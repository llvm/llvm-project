.. title:: clang-tidy - readability-avoid-default-lambda-capture

readability-avoid-default-lambda-capture
========================================

Warns on default lambda captures (e.g. ``[&](){ ... }``, ``[=](){ ... }``).
  
Captures can lead to subtle bugs including dangling references and unnecessary
copies. Writing out the name of the variables being captured reminds programmers
and reviewers to know what is being captured. And knowing is half the battle.

Coding guidelines that recommend against defaulted lambda captures include:

* Item 31 of Effective Modern C++ by Scott Meyers

This check does not lint for variable-length array (VLA) captures. VLAs are not
ISO C++, and it is impossible to explicitly capture them as the syntax does not
exist.

Example
-------

.. code-block:: c++

  #include <iostream>

  class Widget {
    std::vector<std::function<void(int)>> callbacks;
    int                                   widgetId;
    void addCallback(int factoryId) {
      callbacks.emplace_back(
        [&](){
          std::cout << "Widget " << widgetId << " made in factory " << factoryId;
        }
      );
    }
  }

When ``callbacks`` is executed, ``factoryId`` will dangle. Writing the name of
``factoryId`` in the capture list reminds the reader that it is being captured,
which will hopefully lead to the bug being fixed during code review.

.. code-block:: c++

  #include <iostream>

  class Widget {
    std::vector<std::function<void(int)>> callbacks;
    int                                   widgetId;
    void addCallback(int factoryId) {
      callbacks.emplace_back(
        [&factoryId, &widgetId](){ // Why isn't factoryId captured by value??
          std::cout << "Widget " << widgetId << " made in factory " << factoryId;
        }
      );
    }
  }
