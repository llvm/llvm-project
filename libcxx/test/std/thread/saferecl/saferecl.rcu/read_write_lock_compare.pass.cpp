//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// REQUIRES: std-at-least-c++26

// <rcu>

#include <atomic>
#include <cstddef>
#include <rcu>
#include <shared_mutex>
#include <stop_token>
#include <thread>
#include <chrono>
#include <iostream>
#include <print>
#include <string>

#include "make_test_thread.h"
#include "test_macros.h"

constexpr int num_reader = 4;
const std::chrono::seconds test_time(10);


struct alignas(128) MyObject : public std::rcu_obj_base<MyObject> {
  std::string data_;
  std::atomic<size_t>& destruction_count_;

  inline static int instance_count = 0;
  MyObject(std::atomic<size_t>& count) : data_(std::to_string(instance_count++) + " instance very very very long string"), destruction_count_(count) {}
  ~MyObject() {destruction_count_.fetch_add(1, std::memory_order_relaxed);}

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
  std::atomic<size_t> destruction_count = 0;
  MyObject* globalObjRWLock = new MyObject(destruction_count);
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

  auto writer_func = [&globalObjRWLock, &globalObjMutex, &destruction_count](std::stop_token token) {
    int write_count = 0;
    while (!token.stop_requested()) {
      auto newObj = new MyObject(destruction_count);
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
  std::println("Writer thread destruction {} times", destruction_count.load());
}

void test_rcu() {
  std::atomic<size_t> destruction_count = 0;
  std::rcu_domain& dom = std::rcu_default_domain();
  std::atomic<MyObject*> global_obj_rcu = new MyObject(destruction_count);

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

  auto writer_func = [&global_obj_rcu, &destruction_count](std::stop_token token) {
    int write_count = 0;
    while (!token.stop_requested()) {
      auto newObj = new MyObject(destruction_count);
      auto oldObj = global_obj_rcu.exchange(newObj, std::memory_order_relaxed);
      oldObj->retire();
      ++write_count;
      std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
    std::println("RCU Writer thread wrote {} times", write_count);
  };

  auto syncer_func = [&dom](std::stop_token token) {
    while (!token.stop_requested()) {
     std::rcu_barrier(dom);
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
  std::rcu_barrier(dom);
  std::println("RCU collector thread destruction {} times", destruction_count.load());
}

int main(int, char**) {
  std::println("Testing read-write lock:");
  test_read_write_lock();
  std::println("Testing RCU:");
  test_rcu();
  return 1;
}
