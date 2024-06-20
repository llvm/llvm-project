! RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
!ERROR: Character in intrinsic function ichar must have length one
print *, ichar('')
!ERROR: Character in intrinsic function iachar must have length one
print *, iachar('')
print *, ichar('a')
print *, iachar('a')
!PORTABILITY: Character in intrinsic function ichar should have length one
print *, ichar('ab')
!PORTABILITY: Character in intrinsic function iachar should have length one
print *, iachar('ab')
end

