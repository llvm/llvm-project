.. title:: clang-tidy - portability-template-virtual-member-function

portability-template-virtual-member-function
============================================

Finds cases when an uninstantiated virtual member function in a template class causes 
cross-compiler incompatibility.

Upon instantiating a template class, non-virtual member functions don't have to be 
instantiated unless they are used. Virtual member function instantiation on the other hand 
is unspecified and depends on the implementation of the compiler.

In the following snippets the virtual member function is not instantiated by GCC and Clang,
but it is instantiated by MSVC, so while the snippet is accepted by the former compilers,
it is rejected by the latter.

.. code:: c++

    template<typename T>
    struct CrossPlatformError {
        virtual ~CrossPlatformError() = default;
        
        static void used() {}

        virtual void unused() {
            T MSVCError = this;
        };
    };

    int main() {
        CrossPlatformError<int>::used();
        return 0;
    }

Cross-platform projects that need to support MSVC on Windows might see compiler errors
because certain virtual member functions are instantiated, which are not instantiated 
by other compilers on other platforms. This check highlights such virtual member functions.
