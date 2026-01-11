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

#include <rcu>
#include <thread>
#include <chrono>
#include <iostream>
#include <print>
#include <string>

#include "__rcu/rcu_domain.h"
#include "make_test_thread.h"
#include "test_macros.h"

void log(auto start, std::string_view msg) {
  auto now = std::chrono::system_clock::now();
  std::println(std::cout, "[{:%H:%M:%S}] {}", now - start, msg);
}

int loop_num = 10;

struct MyObject : public std::rcu_obj_base<MyObject> {
  std::string data_;

  inline static int instance_count = 0;
  MyObject() : data_(std::to_string(instance_count++) + " instance very very very long string") {}

  ~MyObject() {
    std::println(std::cout, "MyObject {} destructor called", data_);
  }
};

std::atomic<MyObject*> global_obj = nullptr;

int main(int, char**) {
  auto start = std::chrono::system_clock::now();
  auto t1    = support::make_test_thread([start]() {
    std::rcu_domain& dom = std::rcu_default_domain();
    for (int i = 0; i < loop_num; ++i) {
      log(start, "t1: entering rcu read-side critical section " + std::to_string(i));
      dom.lock();
      auto obj = global_obj.load();
      log(start, "t1: reading: " + (obj ? obj->data_ : "nullptr"));
      std::this_thread::sleep_for(std::chrono::seconds(1));
      log(start, "t1: leaving rcu read-side critical section " + std::to_string(i) + " with object " +
                         (obj ? obj->data_ : "nullptr"));
      dom.unlock();
      log(start, "t1: printing all reader states");
      dom.printAllReaderStatesInHex();
    }
  });

  auto t2 = support::make_test_thread([start]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(200));
    std::rcu_domain& dom = std::rcu_default_domain();
    for (int i = 0; i < loop_num; ++i) {
      log(start, "t2: entering rcu read-side critical section " + std::to_string(i));
      dom.lock();
      auto obj = global_obj.load();
      log(start, "t2: reading: " + (obj ? obj->data_ : "nullptr"));
      std::this_thread::sleep_for(std::chrono::seconds(1));
      log(start, "t2: leaving rcu read-side critical section " + std::to_string(i) + " with object " +
                         (obj ? obj->data_ : "nullptr"));
      dom.unlock();
      log(start, "t2: printing all reader states");
      dom.printAllReaderStatesInHex();
    }
  });

  auto t3 = support::make_test_thread([start]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(1100));
    std::rcu_domain& dom = std::rcu_default_domain();
    for (int i = 0; i < loop_num; ++i) {
      log(start, "t3: entering rcu read-side critical section " + std::to_string(i));
      dom.lock();
      auto obj = global_obj.load();
      log(start, "t3: reading: " + (obj ? obj->data_ : "nullptr"));
      std::this_thread::sleep_for(std::chrono::seconds(1));
      log(start, "t3: leaving rcu read-side critical section " + std::to_string(i) + " with object " +
                         (obj ? obj->data_ : "nullptr"));
      dom.unlock();
      log(start, "t3: printing all reader states");
      dom.printAllReaderStatesInHex();
    }
  });
  auto t4 = support::make_test_thread([start]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    std::rcu_domain& dom = std::rcu_default_domain();
    for (int i = 0; i < loop_num; ++i) {
      log(start, "t4: entering rcu read-side critical section " + std::to_string(i));
      dom.lock();
      auto obj = global_obj.load();
      log(start, "t4: reading: " + (obj ? obj->data_ : "nullptr"));
      std::this_thread::sleep_for(std::chrono::seconds(1));
      log(start, "t4: leaving rcu read-side critical section " + std::to_string(i) + " with object " +
                         (obj ? obj->data_ : "nullptr"));
      dom.unlock();
      log(start, "t4: printing all reader states");
      dom.printAllReaderStatesInHex();
    }
  });

  auto t5 = support::make_test_thread([start]() {
    std::this_thread::sleep_for(std::chrono::milliseconds(3000));
    for (int i = 0; i < loop_num; ++i) {
      auto new_obj = new MyObject();

      log(start, "t5: updating global to : " + new_obj->data_);
      auto old = global_obj.exchange(new_obj);
      log(start, "t5: retiring old object " + (old ? old->data_ : "nullptr"));
      old->retire();
      std::this_thread::sleep_for(std::chrono::seconds(1));
    }
  });
  /*
  auto t3 = support::make_test_thread([start]() {
    std::rcu_domain& dom = std::rcu_default_domain();
    log(start, "t3: entering rcu read-side critical section");
    dom.lock();
    log(start, "t3: sleeping");
    std::this_thread::sleep_for(std::chrono::seconds(3));
    log(start, "t3: leaving rcu read-side critical section");
    dom.unlock();
  });  */

  std::this_thread::sleep_for(std::chrono::milliseconds(10));
  for (int i = 0; i < loop_num + 5; ++i) {
    log(start, "t0: calling rcu_synchronize" + std::to_string(i));
    std::rcu_synchronize();
    log(start, "t0: rcu_synchronize returned" + std::to_string(i));
    std::this_thread::sleep_for(std::chrono::seconds(1));
  }

  t1.join();
  t2.join();
  t3.join();
  t4.join();

  return 1;
}
