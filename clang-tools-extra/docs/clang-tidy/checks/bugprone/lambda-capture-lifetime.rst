.. title:: clang-tidy - bugprone-lambda-capture-lifetime

bugprone-lambda-capture-lifetime
================================

Finds lambdas that capture local variables by reference and escape their
local scope by being passed to asynchronous sinks or out-of-scope containers,
potentially causing use-after-free bugs.

Examples:

.. code-block:: c++

  #include <thread>
  #include <vector>
  #include <functional>

  void thread_escape() {
    int local_var = 42;
    // WARNING: 'local_var' is captured by reference but escapes to a thread
    std::thread t([&local_var]() {
      local_var++;
    });
    t.detach();
  } // 'local_var' is destroyed here, causing a use-after-free in the thread.

  std::vector<std::function<void()>> GlobalActions;

  void container_escape() {
    int local_var = 42;
    // WARNING: 'local_var' is captured by reference but escapes to a global container
    GlobalActions.push_back([&local_var]() {
      local_var++;
    });
  } // 'local_var' is destroyed here, but the lambda lives on in GlobalActions.

Options
-------

.. option:: AsyncClasses

   Semicolon-separated list of names of asynchronous classes whose constructors
   are considered escape sinks. Default is ``::std::thread;::std::jthread``.

.. option:: AsyncFunctions

   Semicolon-separated list of names of asynchronous functions that are
   considered escape sinks. Default is ``::std::async``.

.. option:: StorageClasses

   Semicolon-separated list of names of classes that act as long-lived
   storage containers. Default is ``::std::vector``.

.. option:: StorageFunctions

   Semicolon-separated list of names of member functions that store a
   callable into a long-lived container. Default is
   ``push_back;emplace_back;insert;assign``.
