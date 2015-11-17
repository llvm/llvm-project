cat > 1.c << \!
int _start() {
  return 0;
}

int main() {
fn();
return 0;
}
!

cat > fn.c << \!
int fn() {
fn1();
return 0;
}
!

cat > fn2.c << \!
int fn2() {
return 0;
}
!

cat > fn1.c << \!
int fn1() {
fn2();
}
!

gcc -c 1.c fn.c fn2.c fn1.c
ar cr libfn.a fn.o fn2.o
ar cr libfn1.a fn1.o
lld -flavor gnu -target x86_64 -shared -o libfn2.so fn2.o
lld -flavor gnu -target x86_64 1.o libfn.a libfn1.a -o x
lld -flavor gnu -target x86_64 1.o --start-group libfn.a libfn1.a --end-group -o x
lld -flavor gnu -target x86_64 1.o --start-group fn.o fn2.o fn1.o --end-group -o x
lld -flavor gnu -target x86_64 1.o --start-group --whole-archive libfn.a --no-whole-archive libfn1.a --end-group -o x
