// RUN: %clang_cc1 %s -E -embed-dir=%S/Inputs -CC -verify

#if !__has_embed(__FILE__)
#error 1
#elif !__has_embed("media/art.txt")
#error 2
#elif __has_embed("asdkasdjkadsjkdsfjk")
#error 3
#elif __has_embed("asdkasdjkadsjkdsfjk" limit(1))
#error 4
#elif __has_embed("asdkasdjkadsjkdsfjk" suffix(x) limit(1))
#error 5
#elif __has_embed("asdkasdjkadsjkdsfjk" suffix(x) djsakdasjd::xmeow("xD"))
#error 6
#elif !__has_embed(__FILE__ limit(2) prefix(y))
#error 7
#elif !__has_embed(__FILE__ limit(2))
#error 8
#elif __has_embed(__FILE__ dajwdwdjdahwk::meow(x))
#error 9
#elif __has_embed(<media/empty>) != 2
#error 10
#elif __has_embed(<media/empty> limit(0)) != 2
#error 11
#elif __has_embed(<media/art.txt> limit(0)) != 2
#error 12
#elif __has_embed(<media/art.txt> limit(1) clang::offset(1)) != 2
#error 13
#elif !__has_embed(<media/art.txt>)
#error 14
#elif !__has_embed(<media/art.txt> if_empty(meow))
#error 14
#endif
// expected-no-diagnostics
