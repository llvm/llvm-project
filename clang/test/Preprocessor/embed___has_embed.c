// RUN: %clang_cc1 -std=c23 %s -E -embed-dir=%S/Inputs -verify
// expected-no-diagnostics

#if __has_embed(__FILE__) != __STDC_EMBED_FOUND__
#error 1
#elif __has_embed("media/art.txt") != __STDC_EMBED_FOUND__
#error 2
#elif __has_embed("asdkasdjkadsjkdsfjk") != __STDC_EMBED_NOT_FOUND__
#error 3
#elif __has_embed("asdkasdjkadsjkdsfjk" limit(1)) != __STDC_EMBED_NOT_FOUND__
#error 4
#elif __has_embed("asdkasdjkadsjkdsfjk" suffix(x) limit(1)) != __STDC_EMBED_NOT_FOUND__
#error 5
#elif __has_embed("asdkasdjkadsjkdsfjk" suffix(x) djsakdasjd::xmeow("xD")) != __STDC_EMBED_NOT_FOUND__
#error 6
#elif __has_embed(__FILE__ limit(2) prefix(y)) != __STDC_EMBED_FOUND__
#error 7
#elif __has_embed(__FILE__ limit(2)) != __STDC_EMBED_FOUND__
#error 8
// 6.10.1p7, if the search fails or any of the embed parameters in the embed
// parameter sequence specified are not supported by the implementation for the
// #embed directive;
// We don't support one of the embed parameters.
#elif __has_embed(__FILE__ dajwdwdjdahwk::meow(x)) != __STDC_EMBED_NOT_FOUND__
#error 9
#elif __has_embed(<media/empty>) != __STDC_EMBED_EMPTY__
#error 10
// 6.10.1p7: if the search for the resource succeeds and all embed parameters
// in the embed parameter sequence specified are supported by the
// implementation for the #embed directive and the resource is empty
// Limiting to zero characters means the resource is empty.
#elif __has_embed(<media/empty> limit(0)) != __STDC_EMBED_EMPTY__
#error 11
#elif __has_embed(<media/art.txt> limit(0)) != __STDC_EMBED_EMPTY__
#error 12
// FIXME: It's debatable whether this test is correct or not; if we limit the
// file to one character and then offset by one character, the file is empty.
// But if we offset by one character and then limit the file to one character,
// the file is not empty. We do not yet document this extension and so the
// behavior of this might change.
#elif __has_embed(<media/art.txt> limit(1) clang::offset(1)) != __STDC_EMBED_EMPTY__
#error 13
#elif __has_embed(<media/art.txt>) != __STDC_EMBED_FOUND__
#error 14
#elif __has_embed(<media/art.txt> if_empty(meow)) != __STDC_EMBED_FOUND__
#error 14
#endif
