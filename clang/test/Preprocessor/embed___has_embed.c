// RUN: %clang_cc1 -std=c23 %s -E --embed-dir=%S/Inputs -verify
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
// Test that an offset past the end of the file produces an empty file.
#elif __has_embed(<single_byte.txt> clang::offset(1)) != __STDC_EMBED_EMPTY__
#error 13
// Test that we apply the offset before we apply the limit. If we did this in
// the reverse order, this would cause the file to be empty because we would
// have limited it to 1 byte and then offset past it.
#elif __has_embed(<media/art.txt> limit(1) clang::offset(12)) != __STDC_EMBED_FOUND__
#error 14
#elif __has_embed(<media/art.txt>) != __STDC_EMBED_FOUND__
#error 15
#elif __has_embed(<media/art.txt> if_empty(meow)) != __STDC_EMBED_FOUND__
#error 16
#endif

// Ensure that when __has_embed returns true, the file can actually be
// embedded. This was previously failing because the way in which __has_embed
// would search for files was differentl from how #embed would resolve them
// when the file path included relative path markers like `./` or `../`.
#if __has_embed("./embed___has_embed.c") == __STDC_EMBED_FOUND__
unsigned char buffer[] = {
#embed "./embed___has_embed.c"
};
#else
#error 17
#endif
