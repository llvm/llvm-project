//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20, c++23

// <rcu>

#include <atomic>
#include <rcu>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <chrono>
#include <iostream>
#include <print>
#include <string>

#include "__rcu/rcu_domain.h"
#include "make_test_thread.h"
#include "test_macros.h"

constexpr int num_reader = 4;
const std::chrono::seconds test_time(10);

struct alignas(128) MyObject : public std::rcu_obj_base<MyObject> {
  std::string data_;

  inline static int instance_count = 0;
  MyObject() : data_(std::to_string(instance_count++) + " instance very very very long string") {}

  void doWork() {
    auto spin_for = [](std::chrono::microseconds us) {
      auto start = std::chrono::high_resolution_clock::now();
      while (std::chrono::high_resolution_clock::now() - start < us)
        ;
    };
    using namespace std::chrono_literals;
    spin_for(10us);
  }
};


void test_read_write_lock() {
  MyObject* globalObjRWLock = new MyObject();
  std::shared_mutex globalObjMutex;

  std::vector<std::jthread> readers;
  readers.reserve(num_reader);

  auto reader_func = [&globalObjRWLock, &globalObjMutex](std::stop_token token) {
    int read_count = 0;
    while (!token.stop_requested()) {
      std::shared_lock<std::shared_mutex> lock(globalObjMutex);
      globalObjRWLock->doWork();
      ++read_count;
    }
    std::println("Reader thread read {} times", read_count);
  };

  auto writer_func = [&globalObjRWLock, &globalObjMutex](std::stop_token token) {
    int write_count = 0;
    while (!token.stop_requested()) {
      auto newObj = new MyObject();
      std::unique_lock<std::shared_mutex> lock(globalObjMutex);
      auto oldObj     = globalObjRWLock;
      globalObjRWLock = newObj;
      lock.unlock();
      delete oldObj;
      ++write_count;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    std::println("Writer thread wrote {} times", write_count);
  };

  for (int i = 0; i < num_reader; ++i) {
    readers.emplace_back(reader_func);
  }
  std::jthread writer(writer_func);

  std::this_thread::sleep_for(test_time);

  for (auto& reader : readers) {
    reader.request_stop();
  }
  writer.request_stop();
}

void test_rcu() {
  std::rcu_domain& dom = std::rcu_default_domain();
  std::atomic<MyObject*> global_obj_rcu = new MyObject();

  std::vector<std::jthread> readers;
  readers.reserve(num_reader);

  auto reader_func = [&dom, &global_obj_rcu](std::stop_token token) {
    int read_count = 0;
    while (!token.stop_requested()) {
      dom.lock();
      auto obj = global_obj_rcu.load(std::memory_order_relaxed);
      obj->doWork();
      dom.unlock();
      ++read_count;
    }
    std::println("RCU Reader thread read {} times", read_count);
  };

  auto writer_func = [&global_obj_rcu](std::stop_token token) {
    int write_count = 0;
    while (!token.stop_requested()) {
      auto newObj = new MyObject();
      auto oldObj = global_obj_rcu.exchange(newObj, std::memory_order_relaxed);
      oldObj->retire();
      ++write_count;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    std::println("RCU Writer thread wrote {} times", write_count);
  };

  auto syncer_func = [&dom](std::stop_token token) {
    while (!token.stop_requested()) {
      std::rcu_synchronize(dom);
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
  };

  for (int i = 0; i < num_reader; ++i) {
    readers.emplace_back(reader_func);
  }
  std::jthread writer(writer_func);
  std::jthread syncer(syncer_func);

  std::this_thread::sleep_for(test_time);

  for (auto& reader : readers) {
    reader.request_stop();
  }
  writer.request_stop();
  syncer.request_stop();
  std::rcu_synchronize(dom);
}

int main(int, char**) {
  std::println("Testing read-write lock:");
  test_read_write_lock();
  std::println("Testing RCU:");
  test_rcu();
  return 1;
}
