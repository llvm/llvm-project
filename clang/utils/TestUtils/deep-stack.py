#!/usr/bin/env python

def pcall(f, N):
    if N == 0:
        print >>f, '    f(0)'
        return

    print >>f, '    f('
    pcall(f, N - 1)
    print >>f, '     )'

def main():
    f = open('t.c','w')
    print >>f, 'int f(int n) { return n; }'
    print >>f, 'int t() {'
    print >>f, '  return'
    pcall(f, 10000)
    print >>f, '  ;'
    print >>f, '}'

if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(100000)
    main()
