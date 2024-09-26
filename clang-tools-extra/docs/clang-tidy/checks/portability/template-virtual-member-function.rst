.. title:: clang-tidy - portability-template-virtual-member-function

portability-template-virtual-member-function
============================================
Per C++ ``[temp.inst#11]``: "It is unspecified whether or not an implementation 
implicitly instantiates a virtual member function of a class template if the virtual 
member function would not otherwise be instantiated."

In the following snippets the virtual member function is not instantiated by gcc and clang,
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
