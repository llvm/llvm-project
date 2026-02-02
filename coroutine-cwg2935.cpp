#include <array>
#include <cassert>
#include <coroutine>
#include <cstdlib>

enum check_points {
  para_copy_ctor,
  para_dtor,
  promise_ctor,
  promise_dtor,
  get_return_obj,
  task_ctor,
  task_dtor,
  init_suspend,
  init_a_ready,
  init_a_suspend,
  init_a_resume,
  awaiter_ctor,
  awaiter_dtor,
  return_v,
  unhandled,
  fin_suspend
};

using per_test_counts_type = std::array<int, fin_suspend + 1>;

per_test_counts_type per_test_counts{};

void record(check_points cp) {
  // Each checkpoint's executions must be precisely recorded to prevent double
  // free
  ++per_test_counts[cp];
}

void clear() { per_test_counts = per_test_counts_type{}; }

std::array<bool, fin_suspend + 1> checked_cond{};

// Each test will throw an exception at a designated location. After the
// coroutine is invoked, the execution status of all checkpoints will be
// checked, and then switch to the next test. Before throwing an exception,
// record the execution status first.
void throw_c(check_points cp) {
  record(cp);
  // Once that point has been tested, it will not be tested again.
  if (checked_cond[cp] == false) {
    checked_cond[cp] = true;
    throw 0;
  }
}

std::size_t allocate_count = 0;

void *operator new(std::size_t size) {
  ++allocate_count;
  // When the coroutine is invoked, memory allocation is the first operation
  // performed
  if (void *ptr = std::malloc(size)) {
    return ptr;
  }
  std::abort();
}

void operator delete(void *ptr) noexcept {
  // Deallocation is performed last
  --allocate_count;
  std::free(ptr);
}

struct copy_observer {
private:
  copy_observer() = default;

public:
  copy_observer(copy_observer const &) { throw_c(para_copy_ctor); }
  ~copy_observer() { record(para_dtor); }

  static copy_observer get() { return {}; }
};

const auto global_observer = copy_observer::get();

namespace direct_emit {

struct task {
  task() { throw_c(task_ctor); }
  ~task() { record(task_dtor); }
  // In this test, the task should be constructed directly, without copy elision
  task(task const &) = delete;
  struct promise_type {
    promise_type() { throw_c(promise_ctor); }
    ~promise_type() { record(promise_dtor); }
    promise_type(const promise_type &) = delete;
    task get_return_object() {
      throw_c(get_return_obj);
      return {};
    }
    auto initial_suspend() {
      throw_c(init_suspend);
      struct initial_awaiter {
        initial_awaiter() { throw_c(awaiter_ctor); }
        ~initial_awaiter() { record(awaiter_dtor); }
        initial_awaiter(const initial_awaiter &) = delete;
        bool await_ready() {
          throw_c(init_a_ready);
          return false;
        }
        bool await_suspend(std::coroutine_handle<void>) {
          throw_c(init_a_suspend);
          return false;
        }
        void await_resume() {
          // From this point onward, all exceptions are handled by
          // `unhandled_exception` Since the defect of exceptions escaping from
          // `unhandled_exception` remains unresolved (CWG2934), this test only
          // covers the coroutine startup phase. Once CWG2934 is resolved,
          // further tests can be added based on this one.
          record(init_a_resume);
        }
      };

      return initial_awaiter{};
    }
    void return_void() { record(return_v); }
    void unhandled_exception() { record(unhandled); }
    // Note that no exceptions may leak after final_suspend is called, otherwise
    // the behavior is undefined
    std::suspend_never final_suspend() noexcept {
      record(fin_suspend);
      return {};
    }
  };
};

task coro(copy_observer) { co_return; }

void catch_coro() try { coro(global_observer); } catch (...) {
}

// Currently, the conditions at the eight potential exception-throwing points
// need to be checked. More checkpoints can be added after CWG2934 is resolved.
void test() {
  per_test_counts_type e{};
  allocate_count = 0;
  catch_coro();
  e = {
      1, // para_copy_ctor
      0, // para_dtor
      0, // promise_ctor
      0, // promise_dtor
      0, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      0, // promise_dtor
      0, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      1, // task_dtor
      1, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e ==
         per_test_counts); // Clang currently fails starting from this line. If
  // the code you modified causes tests above this line
  // to fail, it indicates that you have broken the
  // correct code and should start over from scratch.
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      1, // task_dtor
      1, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      1, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      1, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      1, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      1, // init_a_suspend
      0, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  // Test for execution without exceptions
  {
    coro(global_observer);
  }
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      1, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      1, // init_a_suspend
      1, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      1, // return_v,
      0, // unhandled,
      1, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  assert(allocate_count == 0);
}

} // namespace direct_emit

namespace no_direct_emit {

struct gro_tag_t {};

struct task {
  task(gro_tag_t) { throw_c(task_ctor); }
  ~task() { record(task_dtor); }
  // In this test, the task should be constructed directly, without copy elision
  task(task const &) = delete;
  struct promise_type {
    promise_type() { throw_c(promise_ctor); }
    ~promise_type() { record(promise_dtor); }
    promise_type(const promise_type &) = delete;
    gro_tag_t get_return_object() {
      throw_c(get_return_obj);
      return {};
    }
    auto initial_suspend() {
      throw_c(init_suspend);
      struct initial_awaiter {
        initial_awaiter() { throw_c(awaiter_ctor); }
        ~initial_awaiter() { record(awaiter_dtor); }
        initial_awaiter(const initial_awaiter &) = delete;
        bool await_ready() {
          throw_c(init_a_ready);
          return false;
        }
        bool await_suspend(std::coroutine_handle<void>) {
          throw_c(init_a_suspend);
          return false;
        }
        void await_resume() {
          // From this point onward, all exceptions are handled by
          // `unhandled_exception` Since the defect of exceptions escaping from
          // `unhandled_exception` remains unresolved (CWG2934), this test only
          // covers the coroutine startup phase. Once CWG2934 is resolved,
          // further tests can be added based on this one.
          record(init_a_resume);
        }
      };

      return initial_awaiter{};
    }
    void return_void() { record(return_v); }
    void unhandled_exception() { record(unhandled); }
    // Note that no exceptions may leak after final_suspend is called, otherwise
    // the behavior is undefined
    std::suspend_never final_suspend() noexcept {
      record(fin_suspend);
      return {};
    }
  };
};

task coro(copy_observer) { co_return; }

void catch_coro() try { coro(global_observer); } catch (...) {
}

// Currently, the conditions at the eight potential exception-throwing points
// need to be checked. More checkpoints can be added after CWG2934 is resolved.
void test() {
  per_test_counts_type e{};
  allocate_count = 0;
  catch_coro();
  e = {
      1, // para_copy_ctor
      0, // para_dtor
      0, // promise_ctor
      0, // promise_dtor
      0, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      0, // promise_dtor
      0, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      0, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      1, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      0, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      1, // init_suspend
      0, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      1, // awaiter_ctor
      0, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      0, // init_a_suspend
      0, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      0, // task_ctor
      0, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      1, // init_a_suspend
      0, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      0, // return_v,
      0, // unhandled,
      0, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  catch_coro();
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      0, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      1, // init_a_suspend
      1, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      1, // return_v,
      0, // unhandled,
      1, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  // Test for execution without exceptions
  {
    coro(global_observer);
  }
  e = {
      2, // para_copy_ctor
      2, // para_dtor
      1, // promise_ctor
      1, // promise_dtor
      1, // get_return_obj
      1, // task_ctor
      1, // task_dtor
      1, // init_suspend
      1, // init_a_ready
      1, // init_a_suspend
      1, // init_a_resume
      1, // awaiter_ctor
      1, // awaiter_dtor
      1, // return_v,
      0, // unhandled,
      1, // fin_suspend
  };
  assert(e == per_test_counts);
  clear();
  assert(allocate_count == 0);
}

} // namespace no_direct_emit

int main() {
  direct_emit::test();
  // clear the global state that records the already thrown points
  checked_cond = {};
  no_direct_emit::test();
}
