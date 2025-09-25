.. title:: clang-tidy - readability-default-lambda-capture

readability-default-lambda-capture
==================================

Warns on default lambda captures (e.g. ``[&](){ ... }``, ``[=](){ ... }``)
  
Captures can lead to subtle bugs including dangling references and unnecessary
copies. Writing out the name of the variables being captured reminds programmers
and reviewers to know what is being captured. And knowing is half the battle.

Coding guidelines that recommend against defaulted lambda captures include:

* Item 31 of Effective Modern C++ by Scott Meyers
* `AUTOSAR C++ Rule A5-1-2 <https://www.mathworks.com/help//releases/
  R2021a/bugfinder/ref/autosarc14rulea512.html>`__

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
``factoryId`` in the capture list makes it easy to review the captures and
detect obvious bugs.

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
