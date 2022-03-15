// RUN: %clang_cc1 -std=c++17 -verify %s

namespace __attribute__(()) A
{
}

namespace A __attribute__(())
{
}

namespace __attribute__(()) [[]] A
{
}

namespace [[]] __attribute__(()) A
{
}

namespace A __attribute__(()) [[]]
{
}

namespace A [[]] __attribute__(())
{
}

namespace [[]] A __attribute__(())
{
}

namespace __attribute__(()) A [[]]
{
}

namespace A::B __attribute__(()) // expected-error{{attributes cannot be specified on a nested namespace definition}}
{
}

namespace __attribute__(()) A::B // expected-error{{attributes cannot be specified on a nested namespace definition}}
{
}
