    .section ".note.ro", "a"
    .p2align 2
    .long 1f - 0f           # name size (not including padding)
    .long 3f - 2f           # desc size (not including padding)
    .long 0x01234567        # type
0:  .asciz "NaMe"	    # name
1:  .p2align 2
2:  .long 0x76543210        # desc
    .long 0x89abcdef
3:  .p2align 2
    .section ".note.rw", "aw"
    .p2align 2
    .long 1f - 0f           # name size (not including padding)
    .long 3f - 2f           # desc size (not including padding)
    .long 0x01234567        # type
0:  .asciz "NaMe"	    # name
1:  .p2align 2
2:  .long 0x76543210        # desc
    .long 0x89abcdef
3:  .p2align 2

