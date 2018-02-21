.. title:: clang-tidy - modernize-use-uncaught-exceptions

modernize-use-uncaught-exceptions
====================================

This check will warn on calls to ``std::uncaught_exception`` and replace them with
calls to ``std::uncaught_exceptions``, since ``std::uncaught_exception`` was deprecated
in C++17.

Below are a few examples of what kind of occurrences will be found and what
they will be replaced with.

.. code-block:: c++

	#define MACRO1 std::uncaught_exception
	#define MACRO2 std::uncaught_exception

	int uncaught_exception() {
		return 0;
	}

	int main() {
		int res;

	  res = uncaught_exception();
	  // No warning, since it is not the deprecated function from namespace std
	  
	  res = MACRO2();
	  // Warning, but will not be replaced
	  
	  res = std::uncaught_exception();
	  // Warning and replaced
	  
	  using std::uncaught_exception;
	  // Warning and replaced
	  
	  res = uncaught_exception();
	  // Warning and replaced
	}

After applying the fixes the code will look like the following:

.. code-block:: c++

	#define MACRO1 std::uncaught_exception
	#define MACRO2 std::uncaught_exception

	int uncaught_exception() {
		return 0;
	}

	int main() {
	  int res;
	  
	  res = uncaught_exception();
	  
	  res = MACRO2();
	  
	  res = std::uncaught_exceptions();
	  
	  using std::uncaught_exceptions;
	  
	  res = uncaught_exceptions();
	}
