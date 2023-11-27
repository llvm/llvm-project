! RUN: %python %S/test_errors.py %s %flang_fc1
character(4) a, b, c, d, e, f
!WARNING: DATA statement value '"abcde"' for 'a' has the wrong length
data a(1:4)/'abcde'/
!WARNING: DATA statement value '"abc"' for 'b' has the wrong length
data b(1:4)/'abc'/
data c/'abcde'/ ! not a substring, conforms
data d/'abc'/ ! not a substring, conforms
!ERROR: Substring must end at 4 or earlier, not 5
data e(1:5)/'abcde'/
!ERROR: Substring must begin at 1 or later, not 0
data f(0:4)/'abcde'/
end
