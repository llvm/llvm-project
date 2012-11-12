// RUN: %clang_cc1 -fsyntax-only -verify -ffreestanding %s
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
