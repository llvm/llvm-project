! RUN: %flang_fc1 -fsyntax-only %s
! This line contains the Latin-1 NBSP (non-breaking space) character '\xa0'
x= 1.
! This line contains the UTF-8 encoding of NBSP ('\xc2' '\xa0')
x=Â 1.
end
