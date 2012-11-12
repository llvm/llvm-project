// RUN: %clang_cc1 -emit-llvm %s -o - | FileCheck %s

// CHECK-NOT: ZN12basic_stringIcEC1Ev
// CHECK: ZN12basic_stringIcED1Ev
// CHECK: ZN12basic_stringIcED1Ev
template<class charT>
class basic_string
{
public:
	basic_string();
	~basic_string();
};

template <class charT>
__attribute__ ((__visibility__("hidden"), __always_inline__)) inline
basic_string<charT>::basic_string()
{
}

template <class charT>
inline
basic_string<charT>::~basic_string()
{
}

typedef basic_string<char> string;

extern template class basic_string<char>;

int main()
{
	string s;
}
