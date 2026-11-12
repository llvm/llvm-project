// RUN: %clang_cc1 -DDIGRAPHS=1 -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -DDIGRAPHS=1 -fno-digraphs -fdigraphs -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -fno-digraphs -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -fdigraphs -fno-digraphs -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -std=c89 -DDIGRAPHS=1 -fdigraphs -fsyntax-only -verify -ffreestanding %s
// RUN: %clang_cc1 -std=c89 -fno-digraphs -fsyntax-only -verify -ffreestanding %s

#if DIGRAPHS

// expected-no-diagnostics
%:include <stdint.h>

    %:ifndef BUFSIZE
     %:define BUFSIZE  512
    %:endif

    void copy(char d<::>, const char s<::>, int len)
    <%
        while (len-- >= 0)
        <%
            d<:len:> = s<:len:>;
        %>
    %>
#else

// expected-error@+1 {{expected identifier or '('}}
%:include <stdint.h>
;
// expected-error@+1 {{expected ')'}} expected-note@+1{{to match this '('}}
void copy(char d<::>);

// expected-error@+1 {{expected function body}}
void copy() <% %>

#endif
