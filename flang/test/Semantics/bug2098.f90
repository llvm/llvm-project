!RUN: %python %S/test_errors.py %s %flang_fc1 -pedantic
!ERROR: DATA statement initializations affect 'a(1_8)' more than once, distinctly
!PORTABILITY: DATA statement initializations affect 'b(1_8)' more than once, identically [-Wmultiple-identical-data]
integer a(2), b(2)
data a(1)/1/
data a(2)/2/
data a(1)/3/
data a(2)/2/
data b(1)/4/
data b(2)/5/
data b(1)/4/
data b(2)/5/
print *, a, b
end
