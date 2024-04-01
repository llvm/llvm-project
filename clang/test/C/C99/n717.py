print("#define M(arg)")

def test(size):
  Prefix = 'U' if size == 8 else 'u'
  # [0x0000 to 0x00A0) excluding [0x0020, 0x007F)
  for val in [val for val in range(0x0000, 0x00A0) if val < 0x0020 or val >= 0x007F]:
     print(f'M(\\{Prefix}{val:0{size}X}) // expected-error {{{{universal character name refers to a control character}}}}')
  print('')
  
  # [0x0020 to 0x007F), excluding 0x0024, 0x0040, and 0x0060
  for val in [val for val in range(0x0020, 0x007F) if val != 0x0024 and val != 0x0040 and val != 0x0060]:
     print(f"M(\\{Prefix}{val:0{size}X}) // expected-error {{{{character '{chr(val)}' cannot be specified by a universal character name}}}}")
  print('')
  
  # [0xD800 to 0xDFFF]
  for val in range(0xD800, 0xDFFF + 1):
    print(f'M(\\{Prefix}{val:0{size}X}) // expected-error {{{{invalid universal character}}}}')
  print('')
  
  # Everything in this range should be accepted, though it may produce a
  # warning diagnostic for things like homoglyphs, whitespace, etc.
  for val in range(0x00A1, 0xD800):
    print(f'M(\\{Prefix}{val:0{size}X})')
  print('')

# Print \u tests
test(4)
# Print \U tests
test(8)

# Validate that the \U characters have the same identity as the \u characters
# within the valid (short) range.
# This is disabled because enabling the test 1) requires using L because u and
# U don't exist until C11, 2) is questionable in terms of value because the
# code points could be different if L isn't using a Unicode encoding, and 3)
# this addition to the test adds 10x the execution time when running the test.
#for val in range(0x00A1, 0xD800):
#  print(f"_Static_assert(L'\\u{val:04X}' == L'\\U{val:08X}', \"\");")
#print('')
