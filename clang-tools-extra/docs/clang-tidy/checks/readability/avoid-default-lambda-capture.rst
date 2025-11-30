.. title:: clang-tidy - readability-avoid-default-lambda-capture

readability-avoid-default-lambda-capture
========================================

Tries to replace default lambda captures (e.g. ``[&](){ ... }``, 
``[=](){ ... }``) with explicit lists of captures.
  
Captures can lead to subtle bugs including dangling references and unnecessary
copies. Writing out the name of the variables being captured reminds programmers
and reviewers about what is being captured.

This check does not warn on variable-length array (VLA) captures. VLAs are not
ISO C++, and it is impossible to explicitly capture them as the syntax for doing
so does not exist.

This check does not provide automatic fixes for macros.

Coding guidelines that recommend against defaulted lambda captures include:

* Item 31 of Effective Modern C++ by Scott Meyers

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
