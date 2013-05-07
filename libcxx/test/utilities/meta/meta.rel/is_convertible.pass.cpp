//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// type_traits

// is_convertible

#include <type_traits>

typedef void Function();
typedef char Array[1];

class NonCopyable {
  NonCopyable(NonCopyable&);
};

int main()
{
    {
    static_assert(( std::is_convertible<void, void>::value), "");
    static_assert(( std::is_convertible<const void, void>::value), "");
    static_assert(( std::is_convertible<void, const void>::value), "");
    static_assert(( std::is_convertible<const void, const void>::value), "");

    static_assert((!std::is_convertible<void, Function>::value), "");
    static_assert((!std::is_convertible<const void, Function>::value), "");

    static_assert((!std::is_convertible<void, Function&>::value), "");
    static_assert((!std::is_convertible<const void, Function&>::value), "");

    static_assert((!std::is_convertible<void, Function*>::value), "");
    static_assert((!std::is_convertible<void, Function* const>::value), "");
    static_assert((!std::is_convertible<const void, Function*>::value), "");
    static_assert((!std::is_convertible<const void, Function*const >::value), "");

    static_assert((!std::is_convertible<void, Array>::value), "");
    static_assert((!std::is_convertible<void, const Array>::value), "");
    static_assert((!std::is_convertible<const void, Array>::value), "");
    static_assert((!std::is_convertible<const void, const Array>::value), "");

    static_assert((!std::is_convertible<void, Array&>::value), "");
    static_assert((!std::is_convertible<void, const Array&>::value), "");
    static_assert((!std::is_convertible<const void, Array&>::value), "");
    static_assert((!std::is_convertible<const void, const Array&>::value), "");

    static_assert((!std::is_convertible<void, char>::value), "");
    static_assert((!std::is_convertible<void, const char>::value), "");
    static_assert((!std::is_convertible<const void, char>::value), "");
    static_assert((!std::is_convertible<const void, const char>::value), "");

    static_assert((!std::is_convertible<void, char&>::value), "");
    static_assert((!std::is_convertible<void, const char&>::value), "");
    static_assert((!std::is_convertible<const void, char&>::value), "");
    static_assert((!std::is_convertible<const void, const char&>::value), "");

    static_assert((!std::is_convertible<void, char*>::value), "");
    static_assert((!std::is_convertible<void, const char*>::value), "");
    static_assert((!std::is_convertible<const void, char*>::value), "");
    static_assert((!std::is_convertible<const void, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<Function, void>::value), "");
    static_assert((!std::is_convertible<Function, const void>::value), "");

    static_assert((!std::is_convertible<Function, Function>::value), "");

    static_assert(( std::is_convertible<Function, Function&>::value), "");

    static_assert(( std::is_convertible<Function, Function*>::value), "");
    static_assert(( std::is_convertible<Function, Function* const>::value), "");

    static_assert((!std::is_convertible<Function, Array>::value), "");
    static_assert((!std::is_convertible<Function, const Array>::value), "");

    static_assert((!std::is_convertible<Function, Array&>::value), "");
    static_assert((!std::is_convertible<Function, const Array&>::value), "");

    static_assert((!std::is_convertible<Function, char>::value), "");
    static_assert((!std::is_convertible<Function, const char>::value), "");

    static_assert((!std::is_convertible<Function, char&>::value), "");
    static_assert((!std::is_convertible<Function, const char&>::value), "");

    static_assert((!std::is_convertible<Function, char*>::value), "");
    static_assert((!std::is_convertible<Function, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<Function&, void>::value), "");
    static_assert((!std::is_convertible<Function&, const void>::value), "");

    static_assert((!std::is_convertible<Function&, Function>::value), "");

    static_assert(( std::is_convertible<Function&, Function&>::value), "");

    static_assert(( std::is_convertible<Function&, Function*>::value), "");
    static_assert(( std::is_convertible<Function&, Function* const>::value), "");

    static_assert((!std::is_convertible<Function&, Array>::value), "");
    static_assert((!std::is_convertible<Function&, const Array>::value), "");

    static_assert((!std::is_convertible<Function&, Array&>::value), "");
    static_assert((!std::is_convertible<Function&, const Array&>::value), "");

    static_assert((!std::is_convertible<Function&, char>::value), "");
    static_assert((!std::is_convertible<Function&, const char>::value), "");

    static_assert((!std::is_convertible<Function&, char&>::value), "");
    static_assert((!std::is_convertible<Function&, const char&>::value), "");

    static_assert((!std::is_convertible<Function&, char*>::value), "");
    static_assert((!std::is_convertible<Function&, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<Function*, void>::value), "");
    static_assert((!std::is_convertible<Function*const, void>::value), "");
    static_assert((!std::is_convertible<Function*, const void>::value), "");
    static_assert((!std::is_convertible<Function*const, const void>::value), "");

    static_assert((!std::is_convertible<Function*, Function>::value), "");
    static_assert((!std::is_convertible<Function*const, Function>::value), "");

    static_assert((!std::is_convertible<Function*, Function&>::value), "");
    static_assert((!std::is_convertible<Function*const, Function&>::value), "");

    static_assert(( std::is_convertible<Function*, Function*>::value), "");
    static_assert(( std::is_convertible<Function*, Function* const>::value), "");
    static_assert(( std::is_convertible<Function*const, Function*>::value), "");
    static_assert(( std::is_convertible<Function*const, Function*const >::value), "");

    static_assert((!std::is_convertible<Function*, Array>::value), "");
    static_assert((!std::is_convertible<Function*, const Array>::value), "");
    static_assert((!std::is_convertible<Function*const, Array>::value), "");
    static_assert((!std::is_convertible<Function*const, const Array>::value), "");

    static_assert((!std::is_convertible<Function*, Array&>::value), "");
    static_assert((!std::is_convertible<Function*, const Array&>::value), "");
    static_assert((!std::is_convertible<Function*const, Array&>::value), "");
    static_assert((!std::is_convertible<Function*const, const Array&>::value), "");

    static_assert((!std::is_convertible<Function*, char>::value), "");
    static_assert((!std::is_convertible<Function*, const char>::value), "");
    static_assert((!std::is_convertible<Function*const, char>::value), "");
    static_assert((!std::is_convertible<Function*const, const char>::value), "");

    static_assert((!std::is_convertible<Function*, char&>::value), "");
    static_assert((!std::is_convertible<Function*, const char&>::value), "");
    static_assert((!std::is_convertible<Function*const, char&>::value), "");
    static_assert((!std::is_convertible<Function*const, const char&>::value), "");

    static_assert((!std::is_convertible<Function*, char*>::value), "");
    static_assert((!std::is_convertible<Function*, const char*>::value), "");
    static_assert((!std::is_convertible<Function*const, char*>::value), "");
    static_assert((!std::is_convertible<Function*const, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<Array, void>::value), "");
    static_assert((!std::is_convertible<const Array, void>::value), "");
    static_assert((!std::is_convertible<Array, const void>::value), "");
    static_assert((!std::is_convertible<const Array, const void>::value), "");

    static_assert((!std::is_convertible<Array, Function>::value), "");
    static_assert((!std::is_convertible<const Array, Function>::value), "");

    static_assert((!std::is_convertible<Array, Function&>::value), "");
    static_assert((!std::is_convertible<const Array, Function&>::value), "");

    static_assert((!std::is_convertible<Array, Function*>::value), "");
    static_assert((!std::is_convertible<Array, Function* const>::value), "");
    static_assert((!std::is_convertible<const Array, Function*>::value), "");
    static_assert((!std::is_convertible<const Array, Function*const >::value), "");

    static_assert((!std::is_convertible<Array, Array>::value), "");
    static_assert((!std::is_convertible<Array, const Array>::value), "");
    static_assert((!std::is_convertible<const Array, Array>::value), "");
    static_assert((!std::is_convertible<const Array, const Array>::value), "");

    static_assert((!std::is_convertible<Array, Array&>::value), "");
    static_assert(( std::is_convertible<Array, const Array&>::value), "");
    static_assert((!std::is_convertible<const Array, Array&>::value), "");
    static_assert(( std::is_convertible<const Array, const Array&>::value), "");

    static_assert((!std::is_convertible<Array, char>::value), "");
    static_assert((!std::is_convertible<Array, const char>::value), "");
    static_assert((!std::is_convertible<const Array, char>::value), "");
    static_assert((!std::is_convertible<const Array, const char>::value), "");

    static_assert((!std::is_convertible<Array, char&>::value), "");
    static_assert((!std::is_convertible<Array, const char&>::value), "");
    static_assert((!std::is_convertible<const Array, char&>::value), "");
    static_assert((!std::is_convertible<const Array, const char&>::value), "");

    static_assert(( std::is_convertible<Array, char*>::value), "");
    static_assert(( std::is_convertible<Array, const char*>::value), "");
    static_assert((!std::is_convertible<const Array, char*>::value), "");
    static_assert(( std::is_convertible<const Array, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<Array&, void>::value), "");
    static_assert((!std::is_convertible<const Array&, void>::value), "");
    static_assert((!std::is_convertible<Array&, const void>::value), "");
    static_assert((!std::is_convertible<const Array&, const void>::value), "");

    static_assert((!std::is_convertible<Array&, Function>::value), "");
    static_assert((!std::is_convertible<const Array&, Function>::value), "");

    static_assert((!std::is_convertible<Array&, Function&>::value), "");
    static_assert((!std::is_convertible<const Array&, Function&>::value), "");

    static_assert((!std::is_convertible<Array&, Function*>::value), "");
    static_assert((!std::is_convertible<Array&, Function* const>::value), "");
    static_assert((!std::is_convertible<const Array&, Function*>::value), "");
    static_assert((!std::is_convertible<const Array&, Function*const >::value), "");

    static_assert((!std::is_convertible<Array&, Array>::value), "");
    static_assert((!std::is_convertible<Array&, const Array>::value), "");
    static_assert((!std::is_convertible<const Array&, Array>::value), "");
    static_assert((!std::is_convertible<const Array&, const Array>::value), "");

    static_assert(( std::is_convertible<Array&, Array&>::value), "");
    static_assert(( std::is_convertible<Array&, const Array&>::value), "");
    static_assert((!std::is_convertible<const Array&, Array&>::value), "");
    static_assert(( std::is_convertible<const Array&, const Array&>::value), "");

    static_assert((!std::is_convertible<Array&, char>::value), "");
    static_assert((!std::is_convertible<Array&, const char>::value), "");
    static_assert((!std::is_convertible<const Array&, char>::value), "");
    static_assert((!std::is_convertible<const Array&, const char>::value), "");

    static_assert((!std::is_convertible<Array&, char&>::value), "");
    static_assert((!std::is_convertible<Array&, const char&>::value), "");
    static_assert((!std::is_convertible<const Array&, char&>::value), "");
    static_assert((!std::is_convertible<const Array&, const char&>::value), "");

    static_assert(( std::is_convertible<Array&, char*>::value), "");
    static_assert(( std::is_convertible<Array&, const char*>::value), "");
    static_assert((!std::is_convertible<const Array&, char*>::value), "");
    static_assert(( std::is_convertible<const Array&, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<char, void>::value), "");
    static_assert((!std::is_convertible<const char, void>::value), "");
    static_assert((!std::is_convertible<char, const void>::value), "");
    static_assert((!std::is_convertible<const char, const void>::value), "");

    static_assert((!std::is_convertible<char, Function>::value), "");
    static_assert((!std::is_convertible<const char, Function>::value), "");

    static_assert((!std::is_convertible<char, Function&>::value), "");
    static_assert((!std::is_convertible<const char, Function&>::value), "");

    static_assert((!std::is_convertible<char, Function*>::value), "");
    static_assert((!std::is_convertible<char, Function* const>::value), "");
    static_assert((!std::is_convertible<const char, Function*>::value), "");
    static_assert((!std::is_convertible<const char, Function*const >::value), "");

    static_assert((!std::is_convertible<char, Array>::value), "");
    static_assert((!std::is_convertible<char, const Array>::value), "");
    static_assert((!std::is_convertible<const char, Array>::value), "");
    static_assert((!std::is_convertible<const char, const Array>::value), "");

    static_assert((!std::is_convertible<char, Array&>::value), "");
    static_assert((!std::is_convertible<char, const Array&>::value), "");
    static_assert((!std::is_convertible<const char, Array&>::value), "");
    static_assert((!std::is_convertible<const char, const Array&>::value), "");

    static_assert(( std::is_convertible<char, char>::value), "");
    static_assert(( std::is_convertible<char, const char>::value), "");
    static_assert(( std::is_convertible<const char, char>::value), "");
    static_assert(( std::is_convertible<const char, const char>::value), "");

    static_assert((!std::is_convertible<char, char&>::value), "");
    static_assert(( std::is_convertible<char, const char&>::value), "");
    static_assert((!std::is_convertible<const char, char&>::value), "");
    static_assert(( std::is_convertible<const char, const char&>::value), "");

    static_assert((!std::is_convertible<char, char*>::value), "");
    static_assert((!std::is_convertible<char, const char*>::value), "");
    static_assert((!std::is_convertible<const char, char*>::value), "");
    static_assert((!std::is_convertible<const char, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<char&, void>::value), "");
    static_assert((!std::is_convertible<const char&, void>::value), "");
    static_assert((!std::is_convertible<char&, const void>::value), "");
    static_assert((!std::is_convertible<const char&, const void>::value), "");

    static_assert((!std::is_convertible<char&, Function>::value), "");
    static_assert((!std::is_convertible<const char&, Function>::value), "");

    static_assert((!std::is_convertible<char&, Function&>::value), "");
    static_assert((!std::is_convertible<const char&, Function&>::value), "");

    static_assert((!std::is_convertible<char&, Function*>::value), "");
    static_assert((!std::is_convertible<char&, Function* const>::value), "");
    static_assert((!std::is_convertible<const char&, Function*>::value), "");
    static_assert((!std::is_convertible<const char&, Function*const >::value), "");

    static_assert((!std::is_convertible<char&, Array>::value), "");
    static_assert((!std::is_convertible<char&, const Array>::value), "");
    static_assert((!std::is_convertible<const char&, Array>::value), "");
    static_assert((!std::is_convertible<const char&, const Array>::value), "");

    static_assert((!std::is_convertible<char&, Array&>::value), "");
    static_assert((!std::is_convertible<char&, const Array&>::value), "");
    static_assert((!std::is_convertible<const char&, Array&>::value), "");
    static_assert((!std::is_convertible<const char&, const Array&>::value), "");

    static_assert(( std::is_convertible<char&, char>::value), "");
    static_assert(( std::is_convertible<char&, const char>::value), "");
    static_assert(( std::is_convertible<const char&, char>::value), "");
    static_assert(( std::is_convertible<const char&, const char>::value), "");

    static_assert(( std::is_convertible<char&, char&>::value), "");
    static_assert(( std::is_convertible<char&, const char&>::value), "");
    static_assert((!std::is_convertible<const char&, char&>::value), "");
    static_assert(( std::is_convertible<const char&, const char&>::value), "");

    static_assert((!std::is_convertible<char&, char*>::value), "");
    static_assert((!std::is_convertible<char&, const char*>::value), "");
    static_assert((!std::is_convertible<const char&, char*>::value), "");
    static_assert((!std::is_convertible<const char&, const char*>::value), "");
    }
    {
    static_assert((!std::is_convertible<char*, void>::value), "");
    static_assert((!std::is_convertible<const char*, void>::value), "");
    static_assert((!std::is_convertible<char*, const void>::value), "");
    static_assert((!std::is_convertible<const char*, const void>::value), "");

    static_assert((!std::is_convertible<char*, Function>::value), "");
    static_assert((!std::is_convertible<const char*, Function>::value), "");

    static_assert((!std::is_convertible<char*, Function&>::value), "");
    static_assert((!std::is_convertible<const char*, Function&>::value), "");

    static_assert((!std::is_convertible<char*, Function*>::value), "");
    static_assert((!std::is_convertible<char*, Function* const>::value), "");
    static_assert((!std::is_convertible<const char*, Function*>::value), "");
    static_assert((!std::is_convertible<const char*, Function*const >::value), "");

    static_assert((!std::is_convertible<char*, Array>::value), "");
    static_assert((!std::is_convertible<char*, const Array>::value), "");
    static_assert((!std::is_convertible<const char*, Array>::value), "");
    static_assert((!std::is_convertible<const char*, const Array>::value), "");

    static_assert((!std::is_convertible<char*, Array&>::value), "");
    static_assert((!std::is_convertible<char*, const Array&>::value), "");
    static_assert((!std::is_convertible<const char*, Array&>::value), "");
    static_assert((!std::is_convertible<const char*, const Array&>::value), "");

    static_assert((!std::is_convertible<char*, char>::value), "");
    static_assert((!std::is_convertible<char*, const char>::value), "");
    static_assert((!std::is_convertible<const char*, char>::value), "");
    static_assert((!std::is_convertible<const char*, const char>::value), "");

    static_assert((!std::is_convertible<char*, char&>::value), "");
    static_assert((!std::is_convertible<char*, const char&>::value), "");
    static_assert((!std::is_convertible<const char*, char&>::value), "");
    static_assert((!std::is_convertible<const char*, const char&>::value), "");

    static_assert(( std::is_convertible<char*, char*>::value), "");
    static_assert(( std::is_convertible<char*, const char*>::value), "");
    static_assert((!std::is_convertible<const char*, char*>::value), "");
    static_assert(( std::is_convertible<const char*, const char*>::value), "");
    }
    {
    static_assert((std::is_convertible<NonCopyable&, NonCopyable&>::value), "");
    static_assert((std::is_convertible<NonCopyable&, const NonCopyable&>::value), "");
    static_assert((std::is_convertible<NonCopyable&, const volatile NonCopyable&>::value), "");
    static_assert((std::is_convertible<NonCopyable&, volatile NonCopyable&>::value), "");
    static_assert((std::is_convertible<const NonCopyable&, const NonCopyable&>::value), "");
    static_assert((std::is_convertible<const NonCopyable&, const volatile NonCopyable&>::value), "");
    static_assert((std::is_convertible<volatile NonCopyable&, const volatile NonCopyable&>::value), "");
    static_assert((std::is_convertible<const volatile NonCopyable&, const volatile NonCopyable&>::value), "");
    static_assert((!std::is_convertible<const NonCopyable&, NonCopyable&>::value), "");
    }
}
