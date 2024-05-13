	.text

	.global	non_split
	.type	non_split,@function
non_split:
	retq
	.size	non_split,. - non_split

	.global	non_split_hidden
        .hidden non_split_hidden
	.type	non_split_hidden,@function
non_split_hidden:
	retq
	.size	non_split_hidden,. - non_split_hidden

	.global non_function_text_symbol
non_function_text_symbol:
	.byte 0x01
	.type	non_function_text_symbol,@STT_OBJECT
	.size	non_function_text_symbol, 1


	.section	.note.GNU-stack,"",@progbits
