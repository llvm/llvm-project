//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

#include <scoped_allocator>
#include <memory>
#include <cassert>

// #include <memory>
//
// template <class Alloc>
// struct allocator_traits
// {
//     typedef Alloc                        allocator_type;
//     typedef typename allocator_type::value_type
//                                          value_type;
//
//     typedef Alloc::pointer | value_type* pointer;
//     typedef Alloc::const_pointer
//           | pointer_traits<pointer>::rebind<const value_type>
//                                          const_pointer;
//     typedef Alloc::void_pointer
//           | pointer_traits<pointer>::rebind<void>
//                                          void_pointer;
//     typedef Alloc::const_void_pointer
//           | pointer_traits<pointer>::rebind<const void>
//                                          const_void_pointer;

template <typename Alloc>
void test_pointer()
{
     typename std::allocator_traits<Alloc>::pointer        vp;
     typename std::allocator_traits<Alloc>::const_pointer cvp;

     ((void)vp); // Prevent unused warning
     ((void)cvp); // Prevent unused warning

     static_assert(std::is_same<bool, decltype( vp ==  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp !=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <=  vp)>::value, "");

     static_assert(std::is_same<bool, decltype( vp == cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp ==  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp != cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp !=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >= cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <= cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <=  vp)>::value, "");

     static_assert(std::is_same<bool, decltype(cvp == cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp != cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >= cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <= cvp)>::value, "");
}

template <typename Alloc>
void test_void_pointer()
{
     typename std::allocator_traits<Alloc>::void_pointer        vp;
     typename std::allocator_traits<Alloc>::const_void_pointer cvp;

     ((void)vp); // Prevent unused warning
     ((void)cvp); // Prevent unused warning

     static_assert(std::is_same<bool, decltype( vp ==  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp !=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <=  vp)>::value, "");

     static_assert(std::is_same<bool, decltype( vp == cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp ==  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp != cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp !=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp >= cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >=  vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <   vp)>::value, "");
     static_assert(std::is_same<bool, decltype( vp <= cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <=  vp)>::value, "");

     static_assert(std::is_same<bool, decltype(cvp == cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp != cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp >= cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <  cvp)>::value, "");
     static_assert(std::is_same<bool, decltype(cvp <= cvp)>::value, "");
}

struct Foo { int x; };

int main(int, char**)
{
    test_pointer<std::scoped_allocator_adaptor<std::allocator<char>>> ();
    test_pointer<std::scoped_allocator_adaptor<std::allocator<int>>> ();
    test_pointer<std::scoped_allocator_adaptor<std::allocator<Foo>>> ();

    test_void_pointer<std::scoped_allocator_adaptor<std::allocator<char>>> ();
    test_void_pointer<std::scoped_allocator_adaptor<std::allocator<int>>> ();
    test_void_pointer<std::scoped_allocator_adaptor<std::allocator<Foo>>> ();

  return 0;
}
