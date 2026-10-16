// RUN: %clang_cc1  -fcxx-exceptions -fexceptions -emit-llvm -debug-info-kind=limited -triple x86_64-apple-darwin10 -std=c++98 %s -o - | FileCheck %s
// The landing pad should have the line number of the closing brace of the function.
// CHECK: ret i32
// CHECK: landingpad {{.*}}
// CHECK-NEXT: !dbg ![[LPAD:[0-9]+]]
// CHECK: ![[LPAD]] = !DILocation(line: 24, column: 1, scope: !{{.*}})

# 1 "/usr/include/c++/4.2.1/vector" 1 3
typedef long unsigned int __darwin_size_t;
typedef __darwin_size_t size_t;
namespace std {
  template<typename _Tp>
  class allocator
  {
  public:
    template<typename _Tp1>
    struct rebind
    { typedef allocator<_Tp1> other; };
    ~allocator() throw() { }
  };
  template<typename _Tp, typename _Alloc>
  struct _Vector_base
  {
    typedef typename _Alloc::template rebind<_Tp>::other _Tp_alloc_type;
    struct _Vector_impl
    {
      _Vector_impl(_Tp_alloc_type const& __a)      { }
    };
    typedef _Alloc allocator_type;
    _Vector_base(const allocator_type& __a)
    : _M_impl(__a)
    {  }
    ~_Vector_base()  {  }
    _Vector_impl _M_impl;
  };
  template<typename _Tp, typename _Alloc = std::allocator<_Tp> >
  class vector
    : protected _Vector_base<_Tp, _Alloc>
  {
    typedef _Vector_base<_Tp, _Alloc> _Base;
  public:
    typedef _Tp value_type;
    typedef size_t size_type;
    typedef _Alloc allocator_type;
    vector(const allocator_type& __a = allocator_type())
      : _Base(__a)
    {      }
    size_type
    push_back(const value_type& __x)
    {}
  };
}
# 10 "main.cpp" 2




int main (int argc, char const *argv[], char const *envp[])
{ // 15
  std::vector<long> longs;
  std::vector<short> shorts;
  for (int i=0; i<12; i++)
    {
      longs.push_back(i);
      shorts.push_back(i);
    }
  return 0; // 23
} // 24
