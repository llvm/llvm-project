//===-------------------------- cxa_demangle.cpp --------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#define _LIBCPP_NO_EXCEPTIONS

#include "__cxxabi_config.h"

#include <vector>
#include <algorithm>
#include <string>
#include <numeric>
#include <cstdlib>
#include <cstring>
#include <cctype>

#ifdef _MSC_VER
// snprintf is implemented in VS 2015
#if _MSC_VER < 1900
#define snprintf _snprintf_s
#endif
#endif

namespace __cxxabiv1
{

namespace
{

enum
{
    unknown_error = -4,
    invalid_args = -3,
    invalid_mangled_name,
    memory_alloc_failure,
    success
};

template <class C>
    const char* parse_type(const char* first, const char* last, C& db);
template <class C>
    const char* parse_encoding(const char* first, const char* last, C& db);
template <class C>
    const char* parse_name(const char* first, const char* last, C& db,
                           bool* ends_with_template_args = 0);
template <class C>
    const char* parse_expression(const char* first, const char* last, C& db);
template <class C>
    const char* parse_template_args(const char* first, const char* last, C& db);
template <class C>
    const char* parse_operator_name(const char* first, const char* last, C& db);
template <class C>
    const char* parse_unqualified_name(const char* first, const char* last, C& db);
template <class C>
    const char* parse_decltype(const char* first, const char* last, C& db);

template <class C>
void
print_stack(const C& db)
{
    fprintf(stderr, "---------\n");
    fprintf(stderr, "names:\n");
    for (auto& s : db.names)
        fprintf(stderr, "{%s#%s}\n", s.first.c_str(), s.second.c_str());
    int i = -1;
    fprintf(stderr, "subs:\n");
    for (auto& v : db.subs)
    {
        if (i >= 0)
            fprintf(stderr, "S%i_ = {", i);
        else
            fprintf(stderr, "S_  = {");
        for (auto& s : v)
            fprintf(stderr, "{%s#%s}", s.first.c_str(), s.second.c_str());
        fprintf(stderr, "}\n");
        ++i;
    }
    fprintf(stderr, "template_param:\n");
    for (auto& t : db.template_param)
    {
        fprintf(stderr, "--\n");
        i = -1;
        for (auto& v : t)
        {
            if (i >= 0)
                fprintf(stderr, "T%i_ = {", i);
            else
                fprintf(stderr, "T_  = {");
            for (auto& s : v)
                fprintf(stderr, "{%s#%s}", s.first.c_str(), s.second.c_str());
            fprintf(stderr, "}\n");
            ++i;
        }
    }
    fprintf(stderr, "---------\n\n");
}

template <class C>
void
print_state(const char* msg, const char* first, const char* last, const C& db)
{
    fprintf(stderr, "%s: ", msg);
    for (; first != last; ++first)
        fprintf(stderr, "%c", *first);
    fprintf(stderr, "\n");
    print_stack(db);
}

// <number> ::= [n] <non-negative decimal integer>

const char*
parse_number(const char* first, const char* last)
{
    if (first != last)
    {
        const char* t = first;
        if (*t == 'n')
            ++t;
        if (t != last)
        {
            if (*t == '0')
            {
                first = t+1;
            }
            else if ('1' <= *t && *t <= '9')
            {
                first = t+1;
                while (first != last && std::isdigit(*first))
                    ++first;
            }
        }
    }
    return first;
}

template <class Float>
struct float_data;

template <>
struct float_data<float>
{
    static const size_t mangled_size = 8;
    static const size_t max_demangled_size = 24;
    static constexpr const char* spec = "%af";
};

constexpr const char* float_data<float>::spec;

template <>
struct float_data<double>
{
    static const size_t mangled_size = 16;
    static const size_t max_demangled_size = 32;
    static constexpr const char* spec = "%a";
};

constexpr const char* float_data<double>::spec;

template <>
struct float_data<long double>
{
#if defined(__mips__) && defined(__mips_n64) || defined(__aarch64__) || \
    defined(__wasm__)
    static const size_t mangled_size = 32;
#elif defined(__arm__) || defined(__mips__) || defined(__hexagon__)
    static const size_t mangled_size = 16;
#else
    static const size_t mangled_size = 20;  // May need to be adjusted to 16 or 24 on other platforms
#endif
    static const size_t max_demangled_size = 40;
    static constexpr const char* spec = "%LaL";
};

constexpr const char* float_data<long double>::spec;

template <class Float, class C>
const char*
parse_floating_number(const char* first, const char* last, C& db)
{
    const size_t N = float_data<Float>::mangled_size;
    if (static_cast<std::size_t>(last - first) > N)
    {
        last = first + N;
        union
        {
            Float value;
            char buf[sizeof(Float)];
        };
        const char* t = first;
        char* e = buf;
        for (; t != last; ++t, ++e)
        {
            if (!isxdigit(*t))
                return first;
            unsigned d1 = isdigit(*t) ? static_cast<unsigned>(*t - '0') :
                                        static_cast<unsigned>(*t - 'a' + 10);
            ++t;
            unsigned d0 = isdigit(*t) ? static_cast<unsigned>(*t - '0') :
                                        static_cast<unsigned>(*t - 'a' + 10);
            *e = static_cast<char>((d1 << 4) + d0);
        }
        if (*t == 'E')
        {
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            std::reverse(buf, e);
#endif
            char num[float_data<Float>::max_demangled_size] = {0};
            int n = snprintf(num, sizeof(num), float_data<Float>::spec, value);
            if (static_cast<std::size_t>(n) >= sizeof(num))
                return first;
            db.names.push_back(typename C::String(num, static_cast<std::size_t>(n)));
            first = t+1;
        }
    }
    return first;
}

// <source-name> ::= <positive length number> <identifier>

template <class C>
const char*
parse_source_name(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        char c = *first;
        if (isdigit(c) && first+1 != last)
        {
            const char* t = first+1;
            size_t n = static_cast<size_t>(c - '0');
            for (c = *t; isdigit(c); c = *t)
            {
                n = n * 10 + static_cast<size_t>(c - '0');
                if (++t == last)
                    return first;
            }
            if (static_cast<size_t>(last - t) >= n)
            {
                typename C::String r(t, n);
                if (r.substr(0, 10) == "_GLOBAL__N")
                    db.names.push_back("(anonymous namespace)");
                else
                    db.names.push_back(std::move(r));
                first = t + n;
            }
        }
    }
    return first;
}

// <substitution> ::= S <seq-id> _
//                ::= S_
// <substitution> ::= Sa # ::std::allocator
// <substitution> ::= Sb # ::std::basic_string
// <substitution> ::= Ss # ::std::basic_string < char,
//                                               ::std::char_traits<char>,
//                                               ::std::allocator<char> >
// <substitution> ::= Si # ::std::basic_istream<char,  std::char_traits<char> >
// <substitution> ::= So # ::std::basic_ostream<char,  std::char_traits<char> >
// <substitution> ::= Sd # ::std::basic_iostream<char, std::char_traits<char> >

template <class C>
const char*
parse_substitution(const char* first, const char* last, C& db)
{
    if (last - first >= 2)
    {
        if (*first == 'S')
        {
            switch (first[1])
            {
            case 'a':
                db.names.push_back("std::allocator");
                first += 2;
                break;
            case 'b':
                db.names.push_back("std::basic_string");
                first += 2;
                break;
            case 's':
                db.names.push_back("std::string");
                first += 2;
                break;
            case 'i':
                db.names.push_back("std::istream");
                first += 2;
                break;
            case 'o':
                db.names.push_back("std::ostream");
                first += 2;
                break;
            case 'd':
                db.names.push_back("std::iostream");
                first += 2;
                break;
            case '_':
                if (!db.subs.empty())
                {
                    for (const auto& n : db.subs.front())
                        db.names.push_back(n);
                    first += 2;
                }
                break;
            default:
                if (std::isdigit(first[1]) || std::isupper(first[1]))
                {
                    size_t sub = 0;
                    const char* t = first+1;
                    if (std::isdigit(*t))
                        sub = static_cast<size_t>(*t - '0');
                    else
                        sub = static_cast<size_t>(*t - 'A') + 10;
                    for (++t; t != last && (std::isdigit(*t) || std::isupper(*t)); ++t)
                    {
                        sub *= 36;
                        if (std::isdigit(*t))
                            sub += static_cast<size_t>(*t - '0');
                        else
                            sub += static_cast<size_t>(*t - 'A') + 10;
                    }
                    if (t == last || *t != '_')
                        return first;
                    ++sub;
                    if (sub < db.subs.size())
                    {
                        for (const auto& n : db.subs[sub])
                            db.names.push_back(n);
                        first = t+1;
                    }
                }
                break;
            }
        }
    }
    return first;
}

// <builtin-type> ::= v    # void
//                ::= w    # wchar_t
//                ::= b    # bool
//                ::= c    # char
//                ::= a    # signed char
//                ::= h    # unsigned char
//                ::= s    # short
//                ::= t    # unsigned short
//                ::= i    # int
//                ::= j    # unsigned int
//                ::= l    # long
//                ::= m    # unsigned long
//                ::= x    # long long, __int64
//                ::= y    # unsigned long long, __int64
//                ::= n    # __int128
//                ::= o    # unsigned __int128
//                ::= f    # float
//                ::= d    # double
//                ::= e    # long double, __float80
//                ::= g    # __float128
//                ::= z    # ellipsis
//                ::= Dd   # IEEE 754r decimal floating point (64 bits)
//                ::= De   # IEEE 754r decimal floating point (128 bits)
//                ::= Df   # IEEE 754r decimal floating point (32 bits)
//                ::= Dh   # IEEE 754r half-precision floating point (16 bits)
//                ::= Di   # char32_t
//                ::= Ds   # char16_t
//                ::= Da   # auto (in dependent new-expressions)
//                ::= Dc   # decltype(auto)
//                ::= Dn   # std::nullptr_t (i.e., decltype(nullptr))
//                ::= u <source-name>    # vendor extended type

template <class C>
const char*
parse_builtin_type(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        switch (*first)
        {
        case 'v':
            db.names.push_back("void");
            ++first;
            break;
        case 'w':
            db.names.push_back("wchar_t");
            ++first;
            break;
        case 'b':
            db.names.push_back("bool");
            ++first;
            break;
        case 'c':
            db.names.push_back("char");
            ++first;
            break;
        case 'a':
            db.names.push_back("signed char");
            ++first;
            break;
        case 'h':
            db.names.push_back("unsigned char");
            ++first;
            break;
        case 's':
            db.names.push_back("short");
            ++first;
            break;
        case 't':
            db.names.push_back("unsigned short");
            ++first;
            break;
        case 'i':
            db.names.push_back("int");
            ++first;
            break;
        case 'j':
            db.names.push_back("unsigned int");
            ++first;
            break;
        case 'l':
            db.names.push_back("long");
            ++first;
            break;
        case 'm':
            db.names.push_back("unsigned long");
            ++first;
            break;
        case 'x':
            db.names.push_back("long long");
            ++first;
            break;
        case 'y':
            db.names.push_back("unsigned long long");
            ++first;
            break;
        case 'n':
            db.names.push_back("__int128");
            ++first;
            break;
        case 'o':
            db.names.push_back("unsigned __int128");
            ++first;
            break;
        case 'f':
            db.names.push_back("float");
            ++first;
            break;
        case 'd':
            db.names.push_back("double");
            ++first;
            break;
        case 'e':
            db.names.push_back("long double");
            ++first;
            break;
        case 'g':
            db.names.push_back("__float128");
            ++first;
            break;
        case 'z':
            db.names.push_back("...");
            ++first;
            break;
        case 'u':
            {
                const char*t = parse_source_name(first+1, last, db);
                if (t != first+1)
                    first = t;
            }
            break;
        case 'D':
            if (first+1 != last)
            {
                switch (first[1])
                {
                case 'd':
                    db.names.push_back("decimal64");
                    first += 2;
                    break;
                case 'e':
                    db.names.push_back("decimal128");
                    first += 2;
                    break;
                case 'f':
                    db.names.push_back("decimal32");
                    first += 2;
                    break;
                case 'h':
                    db.names.push_back("decimal16");
                    first += 2;
                    break;
                case 'i':
                    db.names.push_back("char32_t");
                    first += 2;
                    break;
                case 's':
                    db.names.push_back("char16_t");
                    first += 2;
                    break;
                case 'a':
                    db.names.push_back("auto");
                    first += 2;
                    break;
                case 'c':
                    db.names.push_back("decltype(auto)");
                    first += 2;
                    break;
                case 'n':
                    db.names.push_back("std::nullptr_t");
                    first += 2;
                    break;
                }
            }
            break;
        }
    }
    return first;
}

// <CV-qualifiers> ::= [r] [V] [K]

const char*
parse_cv_qualifiers(const char* first, const char* last, unsigned& cv)
{
    cv = 0;
    if (first != last)
    {
        if (*first == 'r')
        {
            cv |= 4;
            ++first;
        }
        if (*first == 'V')
        {
            cv |= 2;
            ++first;
        }
        if (*first == 'K')
        {
            cv |= 1;
            ++first;
        }
    }
    return first;
}

// <template-param> ::= T_    # first template parameter
//                  ::= T <parameter-2 non-negative number> _

template <class C>
const char*
parse_template_param(const char* first, const char* last, C& db)
{
    if (last - first >= 2)
    {
        if (*first == 'T')
        {
            if (first[1] == '_')
            {
                if (db.template_param.empty())
                    return first;
                if (!db.template_param.back().empty())
                {
                    for (auto& t : db.template_param.back().front())
                        db.names.push_back(t);
                    first += 2;
                }
                else
                {
                    db.names.push_back("T_");
                    first += 2;
                    db.fix_forward_references = true;
                }
            }
            else if (isdigit(first[1]))
            {
                const char* t = first+1;
                size_t sub = static_cast<size_t>(*t - '0');
                for (++t; t != last && isdigit(*t); ++t)
                {
                    sub *= 10;
                    sub += static_cast<size_t>(*t - '0');
                }
                if (t == last || *t != '_' || db.template_param.empty())
                    return first;
                ++sub;
                if (sub < db.template_param.back().size())
                {
                    for (auto& temp : db.template_param.back()[sub])
                        db.names.push_back(temp);
                    first = t+1;
                }
                else
                {
                    db.names.push_back(typename C::String(first, t+1));
                    first = t+1;
                    db.fix_forward_references = true;
                }
            }
        }
    }
    return first;
}

// cc <type> <expression>                               # const_cast<type> (expression)

template <class C>
const char*
parse_const_cast_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'c' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto expr = db.names.back().move_full();
                db.names.pop_back();
                if (db.names.empty())
                    return first;
                db.names.back() = "const_cast<" + db.names.back().move_full() + ">(" + expr + ")";
                first = t1;
            }
        }
    }
    return first;
}

// dc <type> <expression>                               # dynamic_cast<type> (expression)

template <class C>
const char*
parse_dynamic_cast_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto expr = db.names.back().move_full();
                db.names.pop_back();
                if (db.names.empty())
                    return first;
                db.names.back() = "dynamic_cast<" + db.names.back().move_full() + ">(" + expr + ")";
                first = t1;
            }
        }
    }
    return first;
}

// rc <type> <expression>                               # reinterpret_cast<type> (expression)

template <class C>
const char*
parse_reinterpret_cast_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'r' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto expr = db.names.back().move_full();
                db.names.pop_back();
                if (db.names.empty())
                    return first;
                db.names.back() = "reinterpret_cast<" + db.names.back().move_full() + ">(" + expr + ")";
                first = t1;
            }
        }
    }
    return first;
}

// sc <type> <expression>                               # static_cast<type> (expression)

template <class C>
const char*
parse_static_cast_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'c')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto expr = db.names.back().move_full();
                db.names.pop_back();
                db.names.back() = "static_cast<" + db.names.back().move_full() + ">(" + expr + ")";
                first = t1;
            }
        }
    }
    return first;
}

// sp <expression>                                  # pack expansion

template <class C>
const char*
parse_pack_expansion(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'p')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
            first = t;
    }
    return first;
}

// st <type>                                            # sizeof (a type)

template <class C>
const char*
parse_sizeof_type_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 't')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back() = "sizeof (" + db.names.back().move_full() + ")";
            first = t;
        }
    }
    return first;
}

// sz <expr>                                            # sizeof (a expression)

template <class C>
const char*
parse_sizeof_expr_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'z')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back() = "sizeof (" + db.names.back().move_full() + ")";
            first = t;
        }
    }
    return first;
}

// sZ <template-param>                                  # size of a parameter pack

template <class C>
const char*
parse_sizeof_param_pack_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'Z' && first[2] == 'T')
    {
        size_t k0 = db.names.size();
        const char* t = parse_template_param(first+2, last, db);
        size_t k1 = db.names.size();
        if (t != first+2)
        {
            typename C::String tmp("sizeof...(");
            size_t k = k0;
            if (k != k1)
            {
                tmp += db.names[k].move_full();
                for (++k; k != k1; ++k)
                    tmp += ", " + db.names[k].move_full();
            }
            tmp += ")";
            for (; k1 != k0; --k1)
                db.names.pop_back();
            db.names.push_back(std::move(tmp));
            first = t;
        }
    }
    return first;
}

// <function-param> ::= fp <top-level CV-qualifiers> _                                     # L == 0, first parameter
//                  ::= fp <top-level CV-qualifiers> <parameter-2 non-negative number> _   # L == 0, second and later parameters
//                  ::= fL <L-1 non-negative number> p <top-level CV-qualifiers> _         # L > 0, first parameter
//                  ::= fL <L-1 non-negative number> p <top-level CV-qualifiers> <parameter-2 non-negative number> _   # L > 0, second and later parameters

template <class C>
const char*
parse_function_param(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && *first == 'f')
    {
        if (first[1] == 'p')
        {
            unsigned cv;
            const char* t = parse_cv_qualifiers(first+2, last, cv);
            const char* t1 = parse_number(t, last);
            if (t1 != last && *t1 == '_')
            {
                db.names.push_back("fp" + typename C::String(t, t1));
                first = t1+1;
            }
        }
        else if (first[1] == 'L')
        {
            unsigned cv;
            const char* t0 = parse_number(first+2, last);
            if (t0 != last && *t0 == 'p')
            {
                ++t0;
                const char* t = parse_cv_qualifiers(t0, last, cv);
                const char* t1 = parse_number(t, last);
                if (t1 != last && *t1 == '_')
                {
                    db.names.push_back("fp" + typename C::String(t, t1));
                    first = t1+1;
                }
            }
        }
    }
    return first;
}

// sZ <function-param>                                  # size of a function parameter pack

template <class C>
const char*
parse_sizeof_function_param_pack_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 's' && first[1] == 'Z' && first[2] == 'f')
    {
        const char* t = parse_function_param(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back() = "sizeof...(" + db.names.back().move_full() + ")";
            first = t;
        }
    }
    return first;
}

// te <expression>                                      # typeid (expression)
// ti <type>                                            # typeid (type)

template <class C>
const char*
parse_typeid_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 't' && (first[1] == 'e' || first[1] == 'i'))
    {
        const char* t;
        if (first[1] == 'e')
            t = parse_expression(first+2, last, db);
        else
            t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back() = "typeid(" + db.names.back().move_full() + ")";
            first = t;
        }
    }
    return first;
}

// tw <expression>                                      # throw expression

template <class C>
const char*
parse_throw_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 't' && first[1] == 'w')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back() = "throw " + db.names.back().move_full();
            first = t;
        }
    }
    return first;
}

// ds <expression> <expression>                         # expr.*expr

template <class C>
const char*
parse_dot_star_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 's')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto expr = db.names.back().move_full();
                db.names.pop_back();
                db.names.back().first += ".*" + expr;
                first = t1;
            }
        }
    }
    return first;
}

// <simple-id> ::= <source-name> [ <template-args> ]

template <class C>
const char*
parse_simple_id(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        const char* t = parse_source_name(first, last, db);
        if (t != first)
        {
            const char* t1 = parse_template_args(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto args = db.names.back().move_full();
                db.names.pop_back();
                db.names.back().first += std::move(args);
            }
            first = t1;
        }
        else
            first = t;
    }
    return first;
}

// <unresolved-type> ::= <template-param>
//                   ::= <decltype>
//                   ::= <substitution>

template <class C>
const char*
parse_unresolved_type(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        const char* t = first;
        switch (*first)
        {
        case 'T':
          {
            size_t k0 = db.names.size();
            t = parse_template_param(first, last, db);
            size_t k1 = db.names.size();
            if (t != first && k1 == k0 + 1)
            {
                db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                first = t;
            }
            else
            {
                for (; k1 != k0; --k1)
                    db.names.pop_back();
            }
            break;
          }
        case 'D':
            t = parse_decltype(first, last, db);
            if (t != first)
            {
                if (db.names.empty())
                    return first;
                db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                first = t;
            }
            break;
        case 'S':
            t = parse_substitution(first, last, db);
            if (t != first)
                first = t;
            else
            {
                if (last - first > 2 && first[1] == 't')
                {
                    t = parse_unqualified_name(first+2, last, db);
                    if (t != first+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first.insert(0, "std::");
                        db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                        first = t;
                    }
                }
            }
            break;
       }
    }
    return first;
}

// <destructor-name> ::= <unresolved-type>                               # e.g., ~T or ~decltype(f())
//                   ::= <simple-id>                                     # e.g., ~A<2*N>

template <class C>
const char*
parse_destructor_name(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        const char* t = parse_unresolved_type(first, last, db);
        if (t == first)
            t = parse_simple_id(first, last, db);
        if (t != first)
        {
            if (db.names.empty())
                return first;
            db.names.back().first.insert(0, "~");
            first = t;
        }
    }
    return first;
}

// <base-unresolved-name> ::= <simple-id>                                # unresolved name
//          extension     ::= <operator-name>                            # unresolved operator-function-id
//          extension     ::= <operator-name> <template-args>            # unresolved operator template-id
//                        ::= on <operator-name>                         # unresolved operator-function-id
//                        ::= on <operator-name> <template-args>         # unresolved operator template-id
//                        ::= dn <destructor-name>                       # destructor or pseudo-destructor;
//                                                                         # e.g. ~X or ~X<N-1>

template <class C>
const char*
parse_base_unresolved_name(const char* first, const char* last, C& db)
{
    if (last - first >= 2)
    {
        if ((first[0] == 'o' || first[0] == 'd') && first[1] == 'n')
        {
            if (first[0] == 'o')
            {
                const char* t = parse_operator_name(first+2, last, db);
                if (t != first+2)
                {
                    first = parse_template_args(t, last, db);
                    if (first != t)
                    {
                        if (db.names.size() < 2)
                            return first;
                        auto args = db.names.back().move_full();
                        db.names.pop_back();
                        db.names.back().first += std::move(args);
                    }
                }
            }
            else
            {
                const char* t = parse_destructor_name(first+2, last, db);
                if (t != first+2)
                    first = t;
            }
        }
        else
        {
            const char* t = parse_simple_id(first, last, db);
            if (t == first)
            {
                t = parse_operator_name(first, last, db);
                if (t != first)
                {
                    first = parse_template_args(t, last, db);
                    if (first != t)
                    {
                        if (db.names.size() < 2)
                            return first;
                        auto args = db.names.back().move_full();
                        db.names.pop_back();
                        db.names.back().first += std::move(args);
                    }
                }
            }
            else
                first = t;
        }
    }
    return first;
}

// <unresolved-qualifier-level> ::= <simple-id>

template <class C>
const char*
parse_unresolved_qualifier_level(const char* first, const char* last, C& db)
{
    return parse_simple_id(first, last, db);
}

// <unresolved-name>
//  extension        ::= srN <unresolved-type> [<template-args>] <unresolved-qualifier-level>* E <base-unresolved-name>
//                   ::= [gs] <base-unresolved-name>                     # x or (with "gs") ::x
//                   ::= [gs] sr <unresolved-qualifier-level>+ E <base-unresolved-name>  
//                                                                       # A::x, N::y, A<T>::z; "gs" means leading "::"
//                   ::= sr <unresolved-type> <base-unresolved-name>     # T::x / decltype(p)::x
//  extension        ::= sr <unresolved-type> <template-args> <base-unresolved-name>
//                                                                       # T::N::x /decltype(p)::N::x
//  (ignored)        ::= srN <unresolved-type>  <unresolved-qualifier-level>+ E <base-unresolved-name>

template <class C>
const char*
parse_unresolved_name(const char* first, const char* last, C& db)
{
    if (last - first > 2)
    {
        const char* t = first;
        bool global = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            global = true;
            t += 2;
        }
        const char* t2 = parse_base_unresolved_name(t, last, db);
        if (t2 != t)
        {
            if (global)
            {
                if (db.names.empty())
                    return first;
                db.names.back().first.insert(0, "::");
            }
            first = t2;
        }
        else if (last - t > 2 && t[0] == 's' && t[1] == 'r')
        {
            if (t[2] == 'N')
            {
                t += 3;
                const char* t1 = parse_unresolved_type(t, last, db);
                if (t1 == t || t1 == last)
                    return first;
                t = t1;
                t1 = parse_template_args(t, last, db);
                if (t1 != t)
                {
                    if (db.names.size() < 2)
                        return first;
                    auto args = db.names.back().move_full();
                    db.names.pop_back();
                    db.names.back().first += std::move(args);
                    t = t1;
                    if (t == last)
                    {
                        db.names.pop_back();
                        return first;
                    }
                }
                while (*t != 'E')
                {
                    t1 = parse_unresolved_qualifier_level(t, last, db);
                    if (t1 == t || t1 == last || db.names.size() < 2)
                        return first;
                    auto s = db.names.back().move_full();
                    db.names.pop_back();
                    db.names.back().first += "::" + std::move(s);
                    t = t1;
                }
                ++t;
                t1 = parse_base_unresolved_name(t, last, db);
                if (t1 == t)
                {
                    if (!db.names.empty())
                        db.names.pop_back();
                    return first;
                }
                if (db.names.size() < 2)
                    return first;
                auto s = db.names.back().move_full();
                db.names.pop_back();
                db.names.back().first += "::" + std::move(s);
                first = t1;
            }
            else
            {
                t += 2;
                const char* t1 = parse_unresolved_type(t, last, db);
                if (t1 != t)
                {
                    t = t1;
                    t1 = parse_template_args(t, last, db);
                    if (t1 != t)
                    {
                        if (db.names.size() < 2)
                            return first;
                        auto args = db.names.back().move_full();
                        db.names.pop_back();
                        db.names.back().first += std::move(args);
                        t = t1;
                    }
                    t1 = parse_base_unresolved_name(t, last, db);
                    if (t1 == t)
                    {
                        if (!db.names.empty())
                            db.names.pop_back();
                        return first;
                    }
                    if (db.names.size() < 2)
                        return first;
                    auto s = db.names.back().move_full();
                    db.names.pop_back();
                    db.names.back().first += "::" + std::move(s);
                    first = t1;
                }
                else
                {
                    t1 = parse_unresolved_qualifier_level(t, last, db);
                    if (t1 == t || t1 == last)
                        return first;
                    t = t1;
                    if (global)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first.insert(0, "::");
                    }
                    while (*t != 'E')
                    {
                        t1 = parse_unresolved_qualifier_level(t, last, db);
                        if (t1 == t || t1 == last || db.names.size() < 2)
                            return first;
                        auto s = db.names.back().move_full();
                        db.names.pop_back();
                        db.names.back().first += "::" + std::move(s);
                        t = t1;
                    }
                    ++t;
                    t1 = parse_base_unresolved_name(t, last, db);
                    if (t1 == t)
                    {
                        if (!db.names.empty())
                            db.names.pop_back();
                        return first;
                    }
                    if (db.names.size() < 2)
                        return first;
                    auto s = db.names.back().move_full();
                    db.names.pop_back();
                    db.names.back().first += "::" + std::move(s);
                    first = t1;
                }
            }
        }
    }
    return first;
}

// dt <expression> <unresolved-name>                    # expr.name

template <class C>
const char*
parse_dot_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'd' && first[1] == 't')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_unresolved_name(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto name = db.names.back().move_full();
                db.names.pop_back();
                if (db.names.empty())
                    return first;
                db.names.back().first += "." + name;
                first = t1;
            }
        }
    }
    return first;
}

// cl <expression>+ E                                   # call

template <class C>
const char*
parse_call_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 4 && first[0] == 'c' && first[1] == 'l')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (t == last)
                return first;
            if (db.names.empty())
                return first;
            db.names.back().first += db.names.back().second;
            db.names.back().second = typename C::String();
            db.names.back().first.append("(");
            bool first_expr = true;
            while (*t != 'E')
            {
                const char* t1 = parse_expression(t, last, db);
                if (t1 == t || t1 == last)
                    return first;
                if (db.names.empty())
                    return first;
                auto tmp = db.names.back().move_full();
                db.names.pop_back();
                if (!tmp.empty())
                {
                    if (db.names.empty())
                        return first;
                    if (!first_expr)
                    {
                        db.names.back().first.append(", ");
                        first_expr = false;
                    }
                    db.names.back().first.append(tmp);
                }
                t = t1;
            }
            ++t;
            if (db.names.empty())
                return first;
            db.names.back().first.append(")");
            first = t;
        }
    }
    return first;
}

// [gs] nw <expression>* _ <type> E                     # new (expr-list) type
// [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
// [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
// [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
// <initializer> ::= pi <expression>* E                 # parenthesized initialization

template <class C>
const char*
parse_new_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 4)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        if (t[0] == 'n' && (t[1] == 'w' || t[1] == 'a'))
        {
            bool is_array = t[1] == 'a';
            t += 2;
            if (t == last)
                return first;
            bool has_expr_list = false;
            bool first_expr = true;
            while (*t != '_')
            {
                const char* t1 = parse_expression(t, last, db);
                if (t1 == t || t1 == last)
                    return first;
                has_expr_list = true;
                if (!first_expr)
                {
                    if (db.names.empty())
                        return first;
                    auto tmp = db.names.back().move_full();
                    db.names.pop_back();
                    if (!tmp.empty())
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first.append(", ");
                        db.names.back().first.append(tmp);
                        first_expr = false;
                    }
                }
                t = t1;
            }
            ++t;
            const char* t1 = parse_type(t, last, db);
            if (t1 == t || t1 == last)
                return first;
            t = t1;
            bool has_init = false;
            if (last - t >= 3 && t[0] == 'p' && t[1] == 'i')
            {
                t += 2;
                has_init = true;
                first_expr = true;
                while (*t != 'E')
                {
                    t1 = parse_expression(t, last, db);
                    if (t1 == t || t1 == last)
                        return first;
                    if (!first_expr)
                    {
                        if (db.names.empty())
                            return first;
                        auto tmp = db.names.back().move_full();
                        db.names.pop_back();
                        if (!tmp.empty())
                        {
                            if (db.names.empty())
                                return first;
                            db.names.back().first.append(", ");
                            db.names.back().first.append(tmp);
                            first_expr = false;
                        }
                    }
                    t = t1;
                }
            }
            if (*t != 'E')
                return first;
            typename C::String init_list;
            if (has_init)
            {
                if (db.names.empty())
                    return first;
                init_list = db.names.back().move_full();
                db.names.pop_back();
            }
            if (db.names.empty())
                return first;
            auto type = db.names.back().move_full();
            db.names.pop_back();
            typename C::String expr_list;
            if (has_expr_list)
            {
                if (db.names.empty())
                    return first;
                expr_list = db.names.back().move_full();
                db.names.pop_back();
            }
            typename C::String r;
            if (parsed_gs)
                r = "::";
            if (is_array)
                r += "[] ";
            else
                r += " ";
            if (has_expr_list)
                r += "(" + expr_list + ") ";
            r += type;
            if (has_init)
                r += " (" + init_list + ")";
            db.names.push_back(std::move(r));
            first = t+1;
        }
    }
    return first;
}

// cv <type> <expression>                               # conversion with one argument
// cv <type> _ <expression>* E                          # conversion with a different number of arguments

template <class C>
const char*
parse_conversion_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'c' && first[1] == 'v')
    {
        bool try_to_parse_template_args = db.try_to_parse_template_args;
        db.try_to_parse_template_args = false;
        const char* t = parse_type(first+2, last, db);
        db.try_to_parse_template_args = try_to_parse_template_args;
        if (t != first+2 && t != last)
        {
            if (*t != '_')
            {
                const char* t1 = parse_expression(t, last, db);
                if (t1 == t)
                    return first;
                t = t1;
            }
            else
            {
                ++t;
                if (t == last)
                    return first;
                if (*t == 'E')
                    db.names.emplace_back();
                else
                {
                    bool first_expr = true;
                    while (*t != 'E')
                    {
                        const char* t1 = parse_expression(t, last, db);
                        if (t1 == t || t1 == last)
                            return first;
                        if (!first_expr)
                        {
                            if (db.names.empty())
                                return first;
                            auto tmp = db.names.back().move_full();
                            db.names.pop_back();
                            if (!tmp.empty())
                            {
                                if (db.names.empty())
                                    return first;
                                db.names.back().first.append(", ");
                                db.names.back().first.append(tmp);
                                first_expr = false;
                            }
                        }
                        t = t1;
                    }
                }
                ++t;
            }
            if (db.names.size() < 2)
                return first;
            auto tmp = db.names.back().move_full();
            db.names.pop_back();
            db.names.back() = "(" + db.names.back().move_full() + ")(" + tmp + ")";
            first = t;
        }
    }
    return first;
}

// pt <expression> <expression>                    # expr->name

template <class C>
const char*
parse_arrow_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'p' && first[1] == 't')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            const char* t1 = parse_expression(t, last, db);
            if (t1 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto tmp = db.names.back().move_full();
                db.names.pop_back();
                db.names.back().first += "->";
                db.names.back().first += tmp;
                first = t1;
            }
        }
    }
    return first;
}

//  <ref-qualifier> ::= R                   # & ref-qualifier
//  <ref-qualifier> ::= O                   # && ref-qualifier

// <function-type> ::= F [Y] <bare-function-type> [<ref-qualifier>] E

template <class C>
const char*
parse_function_type(const char* first, const char* last, C& db)
{
    if (first != last && *first == 'F')
    {
        const char* t = first+1;
        if (t != last)
        {
            if (*t == 'Y')
            {
                /* extern "C" */
                if (++t == last)
                    return first;
            }
            const char* t1 = parse_type(t, last, db);
            if (t1 != t)
            {
                t = t1;
                typename C::String sig("(");
                int ref_qual = 0;
                while (true)
                {
                    if (t == last)
                    {
                        if (!db.names.empty())
                          db.names.pop_back();
                        return first;
                    }
                    if (*t == 'E')
                    {
                        ++t;
                        break;
                    }
                    if (*t == 'v')
                    {
                        ++t;
                        continue;
                    }
                    if (*t == 'R' && t+1 != last && t[1] == 'E')
                    {
                        ref_qual = 1;
                        ++t;
                        continue;
                    }
                    if (*t == 'O' && t+1 != last && t[1] == 'E')
                    {
                        ref_qual = 2;
                        ++t;
                        continue;
                    }
                    size_t k0 = db.names.size();
                    t1 = parse_type(t, last, db);
                    size_t k1 = db.names.size();
                    if (t1 == t || t1 == last)
                        return first;
                    for (size_t k = k0; k < k1; ++k)
                    {
                        if (sig.size() > 1)
                            sig += ", ";
                        sig += db.names[k].move_full();
                    }
                    for (size_t k = k0; k < k1; ++k)
                        db.names.pop_back();
                    t = t1;
                }
                sig += ")";
                switch (ref_qual)
                {
                case 1:
                    sig += " &";
                    break;
                case 2:
                    sig += " &&";
                    break;
                }
                if (db.names.empty())
                    return first;
                db.names.back().first += " ";
                db.names.back().second.insert(0, sig);
                first = t;
            }
        }
    }
    return first;
}

// <pointer-to-member-type> ::= M <class type> <member type>

template <class C>
const char*
parse_pointer_to_member_type(const char* first, const char* last, C& db)
{
    if (first != last && *first == 'M')
    {
        const char* t = parse_type(first+1, last, db);
        if (t != first+1)
        {
            const char* t2 = parse_type(t, last, db);
            if (t2 != t)
            {
                if (db.names.size() < 2)
                    return first;
                auto func = std::move(db.names.back());
                db.names.pop_back();
                auto class_type = std::move(db.names.back());
                if (!func.second.empty() && func.second.front() == '(')
                {
                    db.names.back().first = std::move(func.first) + "(" + class_type.move_full() + "::*";
                    db.names.back().second = ")" + std::move(func.second);
                }
                else
                {
                    db.names.back().first = std::move(func.first) + " " + class_type.move_full() + "::*";
                    db.names.back().second = std::move(func.second);
                }
                first = t2;
            }
        }
    }
    return first;
}

// <array-type> ::= A <positive dimension number> _ <element type>
//              ::= A [<dimension expression>] _ <element type>

template <class C>
const char*
parse_array_type(const char* first, const char* last, C& db)
{
    if (first != last && *first == 'A' && first+1 != last)
    {
        if (first[1] == '_')
        {
            const char* t = parse_type(first+2, last, db);
            if (t != first+2)
            {
                if (db.names.empty())
                    return first;
                if (db.names.back().second.substr(0, 2) == " [")
                    db.names.back().second.erase(0, 1);
                db.names.back().second.insert(0, " []");
                first = t;
            }
        }
        else if ('1' <= first[1] && first[1] <= '9')
        {
            const char* t = parse_number(first+1, last);
            if (t != last && *t == '_')
            {
                const char* t2 = parse_type(t+1, last, db);
                if (t2 != t+1)
                {
                    if (db.names.empty())
                        return first;
                    if (db.names.back().second.substr(0, 2) == " [")
                        db.names.back().second.erase(0, 1);
                    db.names.back().second.insert(0, " [" + typename C::String(first+1, t) + "]");
                    first = t2;
                }
            }
        }
        else
        {
            const char* t = parse_expression(first+1, last, db);
            if (t != first+1 && t != last && *t == '_')
            {
                const char* t2 = parse_type(++t, last, db);
                if (t2 != t)
                {
                    if (db.names.size() < 2)
                        return first;
                    auto type = std::move(db.names.back());
                    db.names.pop_back();
                    auto expr = std::move(db.names.back());
                    db.names.back().first = std::move(type.first);
                    if (type.second.substr(0, 2) == " [")
                        type.second.erase(0, 1);
                    db.names.back().second = " [" + expr.move_full() + "]" + std::move(type.second);
                    first = t2;
                }
            }
        }
    }
    return first;
}

// <decltype>  ::= Dt <expression> E  # decltype of an id-expression or class member access (C++0x)
//             ::= DT <expression> E  # decltype of an expression (C++0x)

template <class C>
const char*
parse_decltype(const char* first, const char* last, C& db)
{
    if (last - first >= 4 && first[0] == 'D')
    {
        switch (first[1])
        {
        case 't':
        case 'T':
            {
                const char* t = parse_expression(first+2, last, db);
                if (t != first+2 && t != last && *t == 'E')
                {
                    if (db.names.empty())
                        return first;
                    db.names.back() = "decltype(" + db.names.back().move_full() + ")";
                    first = t+1;
                }
            }
            break;
        }
    }
    return first;
}

// extension:
// <vector-type>           ::= Dv <positive dimension number> _
//                                    <extended element type>
//                         ::= Dv [<dimension expression>] _ <element type>
// <extended element type> ::= <element type>
//                         ::= p # AltiVec vector pixel

template <class C>
const char*
parse_vector_type(const char* first, const char* last, C& db)
{
    if (last - first > 3 && first[0] == 'D' && first[1] == 'v')
    {
        if ('1' <= first[2] && first[2] <= '9')
        {
            const char* t = parse_number(first+2, last);
            if (t == last || *t != '_')
                return first;
            const char* num = first + 2;
            size_t sz = static_cast<size_t>(t - num);
            if (++t != last)
            {
                if (*t != 'p')
                {
                    const char* t1 = parse_type(t, last, db);
                    if (t1 != t)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first += " vector[" + typename C::String(num, sz) + "]";
                        first = t1;
                    }
                }
                else
                {
                    ++t;
                    db.names.push_back("pixel vector[" + typename C::String(num, sz) + "]");
                    first = t;
                }
            }
        }
        else
        {
            typename C::String num;
            const char* t1 = first+2;
            if (*t1 != '_')
            {
                const char* t = parse_expression(t1, last, db);
                if (t != t1)
                {
                    if (db.names.empty())
                        return first;
                    num = db.names.back().move_full();
                    db.names.pop_back();
                    t1 = t;
                }
            }
            if (t1 != last && *t1 == '_' && ++t1 != last)
            {
                const char* t = parse_type(t1, last, db);
                if (t != t1)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first += " vector[" + num + "]";
                    first = t;
                }
            }
        }
    }
    return first;
}

// <type> ::= <builtin-type>
//        ::= <function-type>
//        ::= <class-enum-type>
//        ::= <array-type>
//        ::= <pointer-to-member-type>
//        ::= <template-param>
//        ::= <template-template-param> <template-args>
//        ::= <decltype>
//        ::= <substitution>
//        ::= <CV-qualifiers> <type>
//        ::= P <type>        # pointer-to
//        ::= R <type>        # reference-to
//        ::= O <type>        # rvalue reference-to (C++0x)
//        ::= C <type>        # complex pair (C 2000)
//        ::= G <type>        # imaginary (C 2000)
//        ::= Dp <type>       # pack expansion (C++0x)
//        ::= U <source-name> <type>  # vendor extended type qualifier
// extension := U <objc-name> <objc-type>  # objc-type<identifier>
// extension := <vector-type> # <vector-type> starts with Dv

// <objc-name> ::= <k0 number> objcproto <k1 number> <identifier>  # k0 = 9 + <number of digits in k1> + k1
// <objc-type> := <source-name>  # PU<11+>objcproto 11objc_object<source-name> 11objc_object -> id<source-name>

template <class C>
const char*
parse_type(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        switch (*first)
        {
            case 'r':
            case 'V':
            case 'K':
              {
                unsigned cv = 0;
                const char* t = parse_cv_qualifiers(first, last, cv);
                if (t != first)
                {
                    bool is_function = *t == 'F';
                    size_t k0 = db.names.size();
                    const char* t1 = parse_type(t, last, db);
                    size_t k1 = db.names.size();
                    if (t1 != t)
                    {
                        if (is_function)
                            db.subs.pop_back();
                        db.subs.emplace_back(db.names.get_allocator());
                        for (size_t k = k0; k < k1; ++k)
                        {
                            if (is_function)
                            {
                                size_t p = db.names[k].second.size();
                                if (db.names[k].second[p - 2] == '&' &&
                                    db.names[k].second[p - 1] == '&')
                                    p -= 2;
                                else if (db.names[k].second.back() == '&')
                                    p -= 1;
                                if (cv & 1)
                                {
                                    db.names[k].second.insert(p, " const");
                                    p += 6;
                                }
                                if (cv & 2)
                                {
                                    db.names[k].second.insert(p, " volatile");
                                    p += 9;
                                }
                                if (cv & 4)
                                    db.names[k].second.insert(p, " restrict");
                            }
                            else
                            {
                                if (cv & 1)
                                    db.names[k].first.append(" const");
                                if (cv & 2)
                                    db.names[k].first.append(" volatile");
                                if (cv & 4)
                                    db.names[k].first.append(" restrict");
                            }
                            db.subs.back().push_back(db.names[k]);
                        }
                        first = t1;
                    }
                }
              }
                break;
            default:
              {
                const char* t = parse_builtin_type(first, last, db);
                if (t != first)
                {
                    first = t;
                }
                else
                {
                    switch (*first)
                    {
                    case 'A':
                        t = parse_array_type(first, last, db);
                        if (t != first)
                        {
                            if (db.names.empty())
                                return first;
                            first = t;
                            db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                        }
                        break;
                    case 'C':
                        t = parse_type(first+1, last, db);
                        if (t != first+1)
                        {
                            if (db.names.empty())
                                return first;
                            db.names.back().first.append(" complex");
                            first = t;
                            db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                        }
                        break;
                    case 'F':
                        t = parse_function_type(first, last, db);
                        if (t != first)
                        {
                            if (db.names.empty())
                                return first;
                            first = t;
                            db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                        }
                        break;
                    case 'G':
                        t = parse_type(first+1, last, db);
                        if (t != first+1)
                        {
                            if (db.names.empty())
                                return first;
                            db.names.back().first.append(" imaginary");
                            first = t;
                            db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                        }
                        break;
                    case 'M':
                        t = parse_pointer_to_member_type(first, last, db);
                        if (t != first)
                        {
                            if (db.names.empty())
                                return first;
                            first = t;
                            db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                        }
                        break;
                    case 'O':
                      {
                        size_t k0 = db.names.size();
                        t = parse_type(first+1, last, db);
                        size_t k1 = db.names.size();
                        if (t != first+1)
                        {
                            db.subs.emplace_back(db.names.get_allocator());
                            for (size_t k = k0; k < k1; ++k)
                            {
                                if (db.names[k].second.substr(0, 2) == " [")
                                {
                                    db.names[k].first += " (";
                                    db.names[k].second.insert(0, ")");
                                }
                                else if (!db.names[k].second.empty() &&
                                          db.names[k].second.front() == '(')
                                {
                                    db.names[k].first += "(";
                                    db.names[k].second.insert(0, ")");
                                }
                                db.names[k].first.append("&&");
                                db.subs.back().push_back(db.names[k]);
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'P':
                      {
                        size_t k0 = db.names.size();
                        t = parse_type(first+1, last, db);
                        size_t k1 = db.names.size();
                        if (t != first+1)
                        {
                            db.subs.emplace_back(db.names.get_allocator());
                            for (size_t k = k0; k < k1; ++k)
                            {
                                if (db.names[k].second.substr(0, 2) == " [")
                                {
                                    db.names[k].first += " (";
                                    db.names[k].second.insert(0, ")");
                                }
                                else if (!db.names[k].second.empty() &&
                                          db.names[k].second.front() == '(')
                                {
                                    db.names[k].first += "(";
                                    db.names[k].second.insert(0, ")");
                                }
                                if (first[1] != 'U' || db.names[k].first.substr(0, 12) != "objc_object<")
                                {
                                    db.names[k].first.append("*");
                                }
                                else
                                {
                                    db.names[k].first.replace(0, 11, "id");
                                }
                                db.subs.back().push_back(db.names[k]);
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'R':
                      {
                        size_t k0 = db.names.size();
                        t = parse_type(first+1, last, db);
                        size_t k1 = db.names.size();
                        if (t != first+1)
                        {
                            db.subs.emplace_back(db.names.get_allocator());
                            for (size_t k = k0; k < k1; ++k)
                            {
                                if (db.names[k].second.substr(0, 2) == " [")
                                {
                                    db.names[k].first += " (";
                                    db.names[k].second.insert(0, ")");
                                }
                                else if (!db.names[k].second.empty() &&
                                          db.names[k].second.front() == '(')
                                {
                                    db.names[k].first += "(";
                                    db.names[k].second.insert(0, ")");
                                }
                                db.names[k].first.append("&");
                                db.subs.back().push_back(db.names[k]);
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'T':
                      {
                        size_t k0 = db.names.size();
                        t = parse_template_param(first, last, db);
                        size_t k1 = db.names.size();
                        if (t != first)
                        {
                            db.subs.emplace_back(db.names.get_allocator());
                            for (size_t k = k0; k < k1; ++k)
                                db.subs.back().push_back(db.names[k]);
                            if (db.try_to_parse_template_args && k1 == k0+1)
                            {
                                const char* t1 = parse_template_args(t, last, db);
                                if (t1 != t)
                                {
                                    auto args = db.names.back().move_full();
                                    db.names.pop_back();
                                    db.names.back().first += std::move(args);
                                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                    t = t1;
                                }
                            }
                            first = t;
                        }
                        break;
                      }
                    case 'U':
                        if (first+1 != last)
                        {
                            t = parse_source_name(first+1, last, db);
                            if (t != first+1)
                            {
                                const char* t2 = parse_type(t, last, db);
                                if (t2 != t)
                                {
                                    if (db.names.size() < 2)
                                        return first;
                                    auto type = db.names.back().move_full();
                                    db.names.pop_back();
                                    if (db.names.back().first.substr(0, 9) != "objcproto")
                                    {
                                        db.names.back() = type + " " + db.names.back().move_full();
                                    }
                                    else
                                    {
                                        auto proto = db.names.back().move_full();
                                        db.names.pop_back();
                                        t = parse_source_name(proto.data() + 9, proto.data() + proto.size(), db);
                                        if (t != proto.data() + 9)
                                        {
                                            db.names.back() = type + "<" + db.names.back().move_full() + ">";
                                        }
                                        else
                                        {
                                            db.names.push_back(type + " " + proto);
                                        }
                                    }
                                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                    first = t2;
                                }
                            }
                        }
                        break;
                    case 'S':
                        if (first+1 != last && first[1] == 't')
                        {
                            t = parse_name(first, last, db);
                            if (t != first)
                            {
                                if (db.names.empty())
                                    return first;
                                db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                first = t;
                            }
                        }
                        else
                        {
                            t = parse_substitution(first, last, db);
                            if (t != first)
                            {
                                first = t;
                                // Parsed a substitution.  If the substitution is a
                                //  <template-param> it might be followed by <template-args>.
                                t = parse_template_args(first, last, db);
                                if (t != first)
                                {
                                    if (db.names.size() < 2)
                                        return first;
                                    auto template_args = db.names.back().move_full();
                                    db.names.pop_back();
                                    db.names.back().first += template_args;
                                    // Need to create substitution for <template-template-param> <template-args>
                                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                    first = t;
                                }
                            }
                        }
                        break;
                    case 'D':
                        if (first+1 != last)
                        {
                            switch (first[1])
                            {
                            case 'p':
                              {
                                size_t k0 = db.names.size();
                                t = parse_type(first+2, last, db);
                                size_t k1 = db.names.size();
                                if (t != first+2)
                                {
                                    db.subs.emplace_back(db.names.get_allocator());
                                    for (size_t k = k0; k < k1; ++k)
                                        db.subs.back().push_back(db.names[k]);
                                    first = t;
                                    return first;
                                }
                                break;
                              }
                            case 't':
                            case 'T':
                                t = parse_decltype(first, last, db);
                                if (t != first)
                                {
                                    if (db.names.empty())
                                        return first;
                                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                    first = t;
                                    return first;
                                }
                                break;
                            case 'v':
                                t = parse_vector_type(first, last, db);
                                if (t != first)
                                {
                                    if (db.names.empty())
                                        return first;
                                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                    first = t;
                                    return first;
                                }
                                break;
                            }
                        }
                        // drop through
                    default:
                        // must check for builtin-types before class-enum-types to avoid
                        // ambiguities with operator-names
                        t = parse_builtin_type(first, last, db);
                        if (t != first)
                        {
                            first = t;
                        }
                        else
                        {
                            t = parse_name(first, last, db);
                            if (t != first)
                            {
                                if (db.names.empty())
                                    return first;
                                db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                                first = t;
                            }
                        }
                        break;
                    }
              }
                break;
            }
        }
    }
    return first;
}

//   <operator-name>
//                   ::= aa    # &&            
//                   ::= ad    # & (unary)     
//                   ::= an    # &             
//                   ::= aN    # &=            
//                   ::= aS    # =             
//                   ::= cl    # ()            
//                   ::= cm    # ,             
//                   ::= co    # ~             
//                   ::= cv <type>    # (cast)        
//                   ::= da    # delete[]      
//                   ::= de    # * (unary)     
//                   ::= dl    # delete        
//                   ::= dv    # /             
//                   ::= dV    # /=            
//                   ::= eo    # ^             
//                   ::= eO    # ^=            
//                   ::= eq    # ==            
//                   ::= ge    # >=            
//                   ::= gt    # >             
//                   ::= ix    # []            
//                   ::= le    # <=            
//                   ::= li <source-name>  # operator ""
//                   ::= ls    # <<            
//                   ::= lS    # <<=           
//                   ::= lt    # <             
//                   ::= mi    # -             
//                   ::= mI    # -=            
//                   ::= ml    # *             
//                   ::= mL    # *=            
//                   ::= mm    # -- (postfix in <expression> context)           
//                   ::= na    # new[]
//                   ::= ne    # !=            
//                   ::= ng    # - (unary)     
//                   ::= nt    # !             
//                   ::= nw    # new           
//                   ::= oo    # ||            
//                   ::= or    # |             
//                   ::= oR    # |=            
//                   ::= pm    # ->*           
//                   ::= pl    # +             
//                   ::= pL    # +=            
//                   ::= pp    # ++ (postfix in <expression> context)
//                   ::= ps    # + (unary)
//                   ::= pt    # ->            
//                   ::= qu    # ?             
//                   ::= rm    # %             
//                   ::= rM    # %=            
//                   ::= rs    # >>            
//                   ::= rS    # >>=           
//                   ::= v <digit> <source-name>        # vendor extended operator

template <class C>
const char*
parse_operator_name(const char* first, const char* last, C& db)
{
    if (last - first >= 2)
    {
        switch (first[0])
        {
        case 'a':
            switch (first[1])
            {
            case 'a':
                db.names.push_back("operator&&");
                first += 2;
                break;
            case 'd':
            case 'n':
                db.names.push_back("operator&");
                first += 2;
                break;
            case 'N':
                db.names.push_back("operator&=");
                first += 2;
                break;
            case 'S':
                db.names.push_back("operator=");
                first += 2;
                break;
            }
            break;
        case 'c':
            switch (first[1])
            {
            case 'l':
                db.names.push_back("operator()");
                first += 2;
                break;
            case 'm':
                db.names.push_back("operator,");
                first += 2;
                break;
            case 'o':
                db.names.push_back("operator~");
                first += 2;
                break;
            case 'v':
                {
                    bool try_to_parse_template_args = db.try_to_parse_template_args;
                    db.try_to_parse_template_args = false;
                    const char* t = parse_type(first+2, last, db);
                    db.try_to_parse_template_args = try_to_parse_template_args;
                    if (t != first+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first.insert(0, "operator ");
                        db.parsed_ctor_dtor_cv = true;
                        first = t;
                    }
                }
                break;
            }
            break;
        case 'd':
            switch (first[1])
            {
            case 'a':
                db.names.push_back("operator delete[]");
                first += 2;
                break;
            case 'e':
                db.names.push_back("operator*");
                first += 2;
                break;
            case 'l':
                db.names.push_back("operator delete");
                first += 2;
                break;
            case 'v':
                db.names.push_back("operator/");
                first += 2;
                break;
            case 'V':
                db.names.push_back("operator/=");
                first += 2;
                break;
            }
            break;
        case 'e':
            switch (first[1])
            {
            case 'o':
                db.names.push_back("operator^");
                first += 2;
                break;
            case 'O':
                db.names.push_back("operator^=");
                first += 2;
                break;
            case 'q':
                db.names.push_back("operator==");
                first += 2;
                break;
            }
            break;
        case 'g':
            switch (first[1])
            {
            case 'e':
                db.names.push_back("operator>=");
                first += 2;
                break;
            case 't':
                db.names.push_back("operator>");
                first += 2;
                break;
            }
            break;
        case 'i':
            if (first[1] == 'x')
            {
                db.names.push_back("operator[]");
                first += 2;
            }
            break;
        case 'l':
            switch (first[1])
            {
            case 'e':
                db.names.push_back("operator<=");
                first += 2;
                break;
            case 'i':
                {
                    const char* t = parse_source_name(first+2, last, db);
                    if (t != first+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first.insert(0, "operator\"\" ");
                        first = t;
                    }
                }
                break;
            case 's':
                db.names.push_back("operator<<");
                first += 2;
                break;
            case 'S':
                db.names.push_back("operator<<=");
                first += 2;
                break;
            case 't':
                db.names.push_back("operator<");
                first += 2;
                break;
            }
            break;
        case 'm':
            switch (first[1])
            {
            case 'i':
                db.names.push_back("operator-");
                first += 2;
                break;
            case 'I':
                db.names.push_back("operator-=");
                first += 2;
                break;
            case 'l':
                db.names.push_back("operator*");
                first += 2;
                break;
            case 'L':
                db.names.push_back("operator*=");
                first += 2;
                break;
            case 'm':
                db.names.push_back("operator--");
                first += 2;
                break;
            }
            break;
        case 'n':
            switch (first[1])
            {
            case 'a':
                db.names.push_back("operator new[]");
                first += 2;
                break;
            case 'e':
                db.names.push_back("operator!=");
                first += 2;
                break;
            case 'g':
                db.names.push_back("operator-");
                first += 2;
                break;
            case 't':
                db.names.push_back("operator!");
                first += 2;
                break;
            case 'w':
                db.names.push_back("operator new");
                first += 2;
                break;
            }
            break;
        case 'o':
            switch (first[1])
            {
            case 'o':
                db.names.push_back("operator||");
                first += 2;
                break;
            case 'r':
                db.names.push_back("operator|");
                first += 2;
                break;
            case 'R':
                db.names.push_back("operator|=");
                first += 2;
                break;
            }
            break;
        case 'p':
            switch (first[1])
            {
            case 'm':
                db.names.push_back("operator->*");
                first += 2;
                break;
            case 'l':
                db.names.push_back("operator+");
                first += 2;
                break;
            case 'L':
                db.names.push_back("operator+=");
                first += 2;
                break;
            case 'p':
                db.names.push_back("operator++");
                first += 2;
                break;
            case 's':
                db.names.push_back("operator+");
                first += 2;
                break;
            case 't':
                db.names.push_back("operator->");
                first += 2;
                break;
            }
            break;
        case 'q':
            if (first[1] == 'u')
            {
                db.names.push_back("operator?");
                first += 2;
            }
            break;
        case 'r':
            switch (first[1])
            {
            case 'm':
                db.names.push_back("operator%");
                first += 2;
                break;
            case 'M':
                db.names.push_back("operator%=");
                first += 2;
                break;
            case 's':
                db.names.push_back("operator>>");
                first += 2;
                break;
            case 'S':
                db.names.push_back("operator>>=");
                first += 2;
                break;
            }
            break;
        case 'v':
            if (std::isdigit(first[1]))
            {
                const char* t = parse_source_name(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "operator ");
                    first = t;
                }
            }
            break;
        }
    }
    return first;
}

template <class C>
const char*
parse_integer_literal(const char* first, const char* last, const typename C::String& lit, C& db)
{
    const char* t = parse_number(first, last);
    if (t != first && t != last && *t == 'E')
    {
        if (lit.size() > 3)
            db.names.push_back("(" + lit + ")");
        else
            db.names.emplace_back();
        if (*first == 'n')
        {
            db.names.back().first += '-';
            ++first;
        }
        db.names.back().first.append(first, t);
        if (lit.size() <= 3)
            db.names.back().first += lit;
        first = t+1;
    }
    return first;
}

// <expr-primary> ::= L <type> <value number> E                          # integer literal
//                ::= L <type> <value float> E                           # floating literal
//                ::= L <string type> E                                  # string literal
//                ::= L <nullptr type> E                                 # nullptr literal (i.e., "LDnE")
//                ::= L <type> <real-part float> _ <imag-part float> E   # complex floating point literal (C 2000)
//                ::= L <mangled-name> E                                 # external name

template <class C>
const char*
parse_expr_primary(const char* first, const char* last, C& db)
{
    if (last - first >= 4 && *first == 'L')
    {
        switch (first[1])
        {
        case 'w':
            {
            const char* t = parse_integer_literal(first+2, last, "wchar_t", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'b':
            if (first[3] == 'E')
            {
                switch (first[2])
                {
                case '0':
                    db.names.push_back("false");
                    first += 4;
                    break;
                case '1':
                    db.names.push_back("true");
                    first += 4;
                    break;
                }
            }
            break;
        case 'c':
            {
            const char* t = parse_integer_literal(first+2, last, "char", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'a':
            {
            const char* t = parse_integer_literal(first+2, last, "signed char", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'h':
            {
            const char* t = parse_integer_literal(first+2, last, "unsigned char", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 's':
            {
            const char* t = parse_integer_literal(first+2, last, "short", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 't':
            {
            const char* t = parse_integer_literal(first+2, last, "unsigned short", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'i':
            {
            const char* t = parse_integer_literal(first+2, last, "", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'j':
            {
            const char* t = parse_integer_literal(first+2, last, "u", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'l':
            {
            const char* t = parse_integer_literal(first+2, last, "l", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'm':
            {
            const char* t = parse_integer_literal(first+2, last, "ul", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'x':
            {
            const char* t = parse_integer_literal(first+2, last, "ll", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'y':
            {
            const char* t = parse_integer_literal(first+2, last, "ull", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'n':
            {
            const char* t = parse_integer_literal(first+2, last, "__int128", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'o':
            {
            const char* t = parse_integer_literal(first+2, last, "unsigned __int128", db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'f':
            {
            const char* t = parse_floating_number<float>(first+2, last, db);
            if (t != first+2)
                first = t;
            }
            break;
        case 'd':
            {
            const char* t = parse_floating_number<double>(first+2, last, db);
            if (t != first+2)
                first = t;
            }
            break;
         case 'e':
            {
            const char* t = parse_floating_number<long double>(first+2, last, db);
            if (t != first+2)
                first = t;
            }
            break;
        case '_':
            if (first[2] == 'Z')
            {
                const char* t = parse_encoding(first+3, last, db);
                if (t != first+3 && t != last && *t == 'E')
                    first = t+1;
            }
            break;
        case 'T':
            // Invalid mangled name per
            //   http://sourcerytools.com/pipermail/cxx-abi-dev/2011-August/002422.html
            break;
        default:
            {
                // might be named type
                const char* t = parse_type(first+1, last, db);
                if (t != first+1 && t != last)
                {
                    if (*t != 'E')
                    {
                        const char* n = t;
                        for (; n != last && isdigit(*n); ++n)
                            ;
                        if (n != t && n != last && *n == 'E')
                        {
                            if (db.names.empty())
                                return first;
                            db.names.back() = "(" + db.names.back().move_full() + ")" + typename C::String(t, n);
                            first = n+1;
                            break;
                        }
                    }
                    else
                    {
                        first = t+1;
                        break;
                    }
                }
            }
        }
    }
    return first;
}

template <class String>
String
base_name(String& s)
{
    if (s.empty())
        return s;
    if (s == "std::string")
    {
        s = "std::basic_string<char, std::char_traits<char>, std::allocator<char> >";
        return "basic_string";
    }
    if (s == "std::istream")
    {
        s = "std::basic_istream<char, std::char_traits<char> >";
        return "basic_istream";
    }
    if (s == "std::ostream")
    {
        s = "std::basic_ostream<char, std::char_traits<char> >";
        return "basic_ostream";
    }
    if (s == "std::iostream")
    {
        s = "std::basic_iostream<char, std::char_traits<char> >";
        return "basic_iostream";
    }
    const char* const pf = s.data();
    const char* pe = pf + s.size();
    if (pe[-1] == '>')
    {
        unsigned c = 1;
        while (true)
        {
            if (--pe == pf)
                return String();
            if (pe[-1] == '<')
            {
                if (--c == 0)
                {
                    --pe;
                    break;
                }
            }
            else if (pe[-1] == '>')
                ++c;
        }
    }
    if (pe - pf <= 1)
      return String();
    const char* p0 = pe - 1;
    for (; p0 != pf; --p0)
    {
        if (*p0 == ':')
        {
            ++p0;
            break;
        }
    }
    return String(p0, pe);
}

// <ctor-dtor-name> ::= C1    # complete object constructor
//                  ::= C2    # base object constructor
//                  ::= C3    # complete object allocating constructor
//   extension      ::= C5    # ?
//                  ::= D0    # deleting destructor
//                  ::= D1    # complete object destructor
//                  ::= D2    # base object destructor
//   extension      ::= D5    # ?

template <class C>
const char*
parse_ctor_dtor_name(const char* first, const char* last, C& db)
{
    if (last-first >= 2 && !db.names.empty())
    {
        switch (first[0])
        {
        case 'C':
            switch (first[1])
            {
            case '1':
            case '2':
            case '3':
            case '5':
                if (db.names.empty())
                    return first;
                db.names.push_back(base_name(db.names.back().first));
                first += 2;
                db.parsed_ctor_dtor_cv = true;
                break;
            }
            break;
        case 'D':
            switch (first[1])
            {
            case '0':
            case '1':
            case '2':
            case '5':
                if (db.names.empty())
                    return first;
                db.names.push_back("~" + base_name(db.names.back().first));
                first += 2;
                db.parsed_ctor_dtor_cv = true;
                break;
            }
            break;
        }
    }
    return first;
}

// <unnamed-type-name> ::= Ut [ <nonnegative number> ] _
//                     ::= <closure-type-name>
// 
// <closure-type-name> ::= Ul <lambda-sig> E [ <nonnegative number> ] _ 
// 
// <lambda-sig> ::= <parameter type>+  # Parameter types or "v" if the lambda has no parameters

template <class C>
const char*
parse_unnamed_type_name(const char* first, const char* last, C& db)
{
    if (last - first > 2 && first[0] == 'U')
    {
        char type = first[1];
        switch (type)
        {
        case 't':
          {
            db.names.push_back(typename C::String("'unnamed"));
            const char* t0 = first+2;
            if (t0 == last)
            {
                db.names.pop_back();
                return first;
            }
            if (std::isdigit(*t0))
            {
                const char* t1 = t0 + 1;
                while (t1 != last && std::isdigit(*t1))
                    ++t1;
                db.names.back().first.append(t0, t1);
                t0 = t1;
            }
            db.names.back().first.push_back('\'');
            if (t0 == last || *t0 != '_')
            {
                db.names.pop_back();
                return first;
            }
            first = t0 + 1;
          }
            break;
        case 'l':
          {
            db.names.push_back(typename C::String("'lambda'("));
            const char* t0 = first+2;
            if (first[2] == 'v')
            {
                db.names.back().first += ')';
                ++t0;
            }
            else
            {
                const char* t1 = parse_type(t0, last, db);
                if (t1 == t0)
                {
                    if(!db.names.empty())
                        db.names.pop_back();
                    return first;
                }
                if (db.names.size() < 2)
                    return first;
                auto tmp = db.names.back().move_full();
                db.names.pop_back();
                db.names.back().first.append(tmp);
                t0 = t1;
                while (true)
                {
                    t1 = parse_type(t0, last, db);
                    if (t1 == t0)
                        break;
                    if (db.names.size() < 2)
                        return first;
                    tmp = db.names.back().move_full();
                    db.names.pop_back();
                    if (!tmp.empty())
                    {
                        db.names.back().first.append(", ");
                        db.names.back().first.append(tmp);
                    }
                    t0 = t1;
                }
                if(db.names.empty())
                  return first;
                db.names.back().first.append(")");
            }
            if (t0 == last || *t0 != 'E')
            {
              if (!db.names.empty())
                db.names.pop_back();
              return first;
            }
            ++t0;
            if (t0 == last)
            {
                if(!db.names.empty())
                  db.names.pop_back();
                return first;
            }
            if (std::isdigit(*t0))
            {
                const char* t1 = t0 + 1;
                while (t1 != last && std::isdigit(*t1))
                    ++t1;
                db.names.back().first.insert(db.names.back().first.begin()+7, t0, t1);
                t0 = t1;
            }
            if (t0 == last || *t0 != '_')
            {
                if(!db.names.empty())
                  db.names.pop_back();
                return first;
            }
            first = t0 + 1;
          }
            break;
        }
    }
    return first;
}

// <unqualified-name> ::= <operator-name>
//                    ::= <ctor-dtor-name>
//                    ::= <source-name>   
//                    ::= <unnamed-type-name>

template <class C>
const char*
parse_unqualified_name(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        const char* t;
        switch (*first)
        {
        case 'C':
        case 'D':
            t = parse_ctor_dtor_name(first, last, db);
            if (t != first)
                first = t;
            break;
        case 'U':
            t = parse_unnamed_type_name(first, last, db);
            if (t != first)
                first = t;
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            t = parse_source_name(first, last, db);
            if (t != first)
                first = t;
            break;
        default:
            t = parse_operator_name(first, last, db);
            if (t != first)
                first = t;
            break;
        };
    }
    return first;
}

// <unscoped-name> ::= <unqualified-name>
//                 ::= St <unqualified-name>   # ::std::
// extension       ::= StL<unqualified-name>

template <class C>
const char*
parse_unscoped_name(const char* first, const char* last, C& db)
{
    if (last - first >= 2)
    {
        const char* t0 = first;
        bool St = false;
        if (first[0] == 'S' && first[1] == 't')
        {
            t0 += 2;
            St = true;
            if (t0 != last && *t0 == 'L')
                ++t0;
        }
        const char* t1 = parse_unqualified_name(t0, last, db);
        if (t1 != t0)
        {
            if (St)
            {
                if (db.names.empty())
                    return first;
                db.names.back().first.insert(0, "std::");
            }
            first = t1;
        }
    }
    return first;
}

// at <type>                                            # alignof (a type)

template <class C>
const char*
parse_alignof_type(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'a' && first[1] == 't')
    {
        const char* t = parse_type(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back().first = "alignof (" + db.names.back().move_full() + ")";
            first = t;
        }
    }
    return first;
}

// az <expression>                                            # alignof (a expression)

template <class C>
const char*
parse_alignof_expr(const char* first, const char* last, C& db)
{
    if (last - first >= 3 && first[0] == 'a' && first[1] == 'z')
    {
        const char* t = parse_expression(first+2, last, db);
        if (t != first+2)
        {
            if (db.names.empty())
                return first;
            db.names.back().first = "alignof (" + db.names.back().move_full() + ")";
            first = t;
        }
    }
    return first;
}

template <class C>
const char*
parse_noexcept_expression(const char* first, const char* last, C& db)
{
    const char* t1 = parse_expression(first, last, db);
    if (t1 != first)
    {
        if (db.names.empty())
            return first;
        db.names.back().first =  "noexcept (" + db.names.back().move_full() + ")";
        first = t1;
    }
    return first;
}

template <class C>
const char*
parse_prefix_expression(const char* first, const char* last, const typename C::String& op, C& db)
{
    const char* t1 = parse_expression(first, last, db);
    if (t1 != first)
    {
        if (db.names.empty())
            return first;
        db.names.back().first =  op + "(" + db.names.back().move_full() + ")";
        first = t1;
    }
    return first;
}

template <class C>
const char*
parse_binary_expression(const char* first, const char* last, const typename C::String& op, C& db)
{
    const char* t1 = parse_expression(first, last, db);
    if (t1 != first)
    {
        const char* t2 = parse_expression(t1, last, db);
        if (t2 != t1)
        {
            if (db.names.size() < 2)
                return first;
            auto op2 = db.names.back().move_full();
            db.names.pop_back();
            auto op1 = db.names.back().move_full();
            auto& nm = db.names.back().first;
            nm.clear();
            if (op == ">")
                nm += '(';
            nm += "(" + op1 + ") " + op + " (" + op2 + ")";
            if (op == ">")
                nm += ')';
            first = t2;
        }
        else if(!db.names.empty())
            db.names.pop_back();
    }
    return first;
}

// <expression> ::= <unary operator-name> <expression>
//              ::= <binary operator-name> <expression> <expression>
//              ::= <ternary operator-name> <expression> <expression> <expression>
//              ::= cl <expression>+ E                                   # call
//              ::= cv <type> <expression>                               # conversion with one argument
//              ::= cv <type> _ <expression>* E                          # conversion with a different number of arguments
//              ::= [gs] nw <expression>* _ <type> E                     # new (expr-list) type
//              ::= [gs] nw <expression>* _ <type> <initializer>         # new (expr-list) type (init)
//              ::= [gs] na <expression>* _ <type> E                     # new[] (expr-list) type
//              ::= [gs] na <expression>* _ <type> <initializer>         # new[] (expr-list) type (init)
//              ::= [gs] dl <expression>                                 # delete expression
//              ::= [gs] da <expression>                                 # delete[] expression
//              ::= pp_ <expression>                                     # prefix ++
//              ::= mm_ <expression>                                     # prefix --
//              ::= ti <type>                                            # typeid (type)
//              ::= te <expression>                                      # typeid (expression)
//              ::= dc <type> <expression>                               # dynamic_cast<type> (expression)
//              ::= sc <type> <expression>                               # static_cast<type> (expression)
//              ::= cc <type> <expression>                               # const_cast<type> (expression)
//              ::= rc <type> <expression>                               # reinterpret_cast<type> (expression)
//              ::= st <type>                                            # sizeof (a type)
//              ::= sz <expression>                                      # sizeof (an expression)
//              ::= at <type>                                            # alignof (a type)
//              ::= az <expression>                                      # alignof (an expression)
//              ::= nx <expression>                                      # noexcept (expression)
//              ::= <template-param>
//              ::= <function-param>
//              ::= dt <expression> <unresolved-name>                    # expr.name
//              ::= pt <expression> <unresolved-name>                    # expr->name
//              ::= ds <expression> <expression>                         # expr.*expr
//              ::= sZ <template-param>                                  # size of a parameter pack
//              ::= sZ <function-param>                                  # size of a function parameter pack
//              ::= sp <expression>                                      # pack expansion
//              ::= tw <expression>                                      # throw expression
//              ::= tr                                                   # throw with no operand (rethrow)
//              ::= <unresolved-name>                                    # f(p), N::f(p), ::f(p),
//                                                                       # freestanding dependent name (e.g., T::x),
//                                                                       # objectless nonstatic member reference
//              ::= <expr-primary>

template <class C>
const char*
parse_expression(const char* first, const char* last, C& db)
{
    if (last - first >= 2)
    {
        const char* t = first;
        bool parsed_gs = false;
        if (last - first >= 4 && t[0] == 'g' && t[1] == 's')
        {
            t += 2;
            parsed_gs = true;
        }
        switch (*t)
        {
        case 'L':
            first = parse_expr_primary(first, last, db);
            break;
        case 'T':
            first = parse_template_param(first, last, db);
            break;
        case 'f':
            first = parse_function_param(first, last, db);
            break;
        case 'a':
            switch (t[1])
            {
            case 'a':
                t = parse_binary_expression(first+2, last, "&&", db);
                if (t != first+2)
                    first = t;
                break;
            case 'd':
                t = parse_prefix_expression(first+2, last, "&", db);
                if (t != first+2)
                    first = t;
                break;
            case 'n':
                t = parse_binary_expression(first+2, last, "&", db);
                if (t != first+2)
                    first = t;
                break;
            case 'N':
                t = parse_binary_expression(first+2, last, "&=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'S':
                t = parse_binary_expression(first+2, last, "=", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                first = parse_alignof_type(first, last, db);
                break;
            case 'z':
                first = parse_alignof_expr(first, last, db);
                break;
            }
            break;
        case 'c':
            switch (t[1])
            {
            case 'c':
                first = parse_const_cast_expr(first, last, db);
                break;
            case 'l':
                first = parse_call_expr(first, last, db);
                break;
            case 'm':
                t = parse_binary_expression(first+2, last, ",", db);
                if (t != first+2)
                    first = t;
                break;
            case 'o':
                t = parse_prefix_expression(first+2, last, "~", db);
                if (t != first+2)
                    first = t;
                break;
            case 'v':
                first = parse_conversion_expr(first, last, db);
                break;
            }
            break;
        case 'd':
            switch (t[1])
            {
            case 'a':
                {
                    const char* t1 = parse_expression(t+2, last, db);
                    if (t1 != t+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first = (parsed_gs ? typename C::String("::") : typename C::String()) +
                                          "delete[] " + db.names.back().move_full();
                        first = t1;
                    }
                }
                break;
            case 'c':
                first = parse_dynamic_cast_expr(first, last, db);
                break;
            case 'e':
                t = parse_prefix_expression(first+2, last, "*", db);
                if (t != first+2)
                    first = t;
                break;
            case 'l':
                {
                    const char* t1 = parse_expression(t+2, last, db);
                    if (t1 != t+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back().first = (parsed_gs ? typename C::String("::") : typename C::String()) +
                                          "delete " + db.names.back().move_full();
                        first = t1;
                    }
                }
                break;
            case 'n':
                return parse_unresolved_name(first, last, db);
            case 's':
                first = parse_dot_star_expr(first, last, db);
                break;
            case 't':
                first = parse_dot_expr(first, last, db);
                break;
            case 'v':
                t = parse_binary_expression(first+2, last, "/", db);
                if (t != first+2)
                    first = t;
                break;
            case 'V':
                t = parse_binary_expression(first+2, last, "/=", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'e':
            switch (t[1])
            {
            case 'o':
                t = parse_binary_expression(first+2, last, "^", db);
                if (t != first+2)
                    first = t;
                break;
            case 'O':
                t = parse_binary_expression(first+2, last, "^=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'q':
                t = parse_binary_expression(first+2, last, "==", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'g':
            switch (t[1])
            {
            case 'e':
                t = parse_binary_expression(first+2, last, ">=", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                t = parse_binary_expression(first+2, last, ">", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'i':
            if (t[1] == 'x')
            {
                const char* t1 = parse_expression(first+2, last, db);
                if (t1 != first+2)
                {
                    const char* t2 = parse_expression(t1, last, db);
                    if (t2 != t1)
                    {
                        if (db.names.size() < 2)
                            return first;
                        auto op2 = db.names.back().move_full();
                        db.names.pop_back();
                        auto op1 = db.names.back().move_full();
                        db.names.back() = "(" + op1 + ")[" + op2 + "]";
                        first = t2;
                    }
                    else if (!db.names.empty())
                        db.names.pop_back();
                }
            }
            break;
        case 'l':
            switch (t[1])
            {
            case 'e':
                t = parse_binary_expression(first+2, last, "<=", db);
                if (t != first+2)
                    first = t;
                break;
            case 's':
                t = parse_binary_expression(first+2, last, "<<", db);
                if (t != first+2)
                    first = t;
                break;
            case 'S':
                t = parse_binary_expression(first+2, last, "<<=", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                t = parse_binary_expression(first+2, last, "<", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'm':
            switch (t[1])
            {
            case 'i':
                t = parse_binary_expression(first+2, last, "-", db);
                if (t != first+2)
                    first = t;
                break;
            case 'I':
                t = parse_binary_expression(first+2, last, "-=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'l':
                t = parse_binary_expression(first+2, last, "*", db);
                if (t != first+2)
                    first = t;
                break;
            case 'L':
                t = parse_binary_expression(first+2, last, "*=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'm':
                if (first+2 != last && first[2] == '_')
                {
                    t = parse_prefix_expression(first+3, last, "--", db);
                    if (t != first+3)
                        first = t;
                }
                else
                {
                    const char* t1 = parse_expression(first+2, last, db);
                    if (t1 != first+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back() = "(" + db.names.back().move_full() + ")--";
                        first = t1;
                    }
                }
                break;
            }
            break;
        case 'n':
            switch (t[1])
            {
            case 'a':
            case 'w':
                first = parse_new_expr(first, last, db);
                break;
            case 'e':
                t = parse_binary_expression(first+2, last, "!=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'g':
                t = parse_prefix_expression(first+2, last, "-", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                t = parse_prefix_expression(first+2, last, "!", db);
                if (t != first+2)
                    first = t;
                break;
            case 'x':
                t = parse_noexcept_expression(first+2, last, db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'o':
            switch (t[1])
            {
            case 'n':
                return parse_unresolved_name(first, last, db);
            case 'o':
                t = parse_binary_expression(first+2, last, "||", db);
                if (t != first+2)
                    first = t;
                break;
            case 'r':
                t = parse_binary_expression(first+2, last, "|", db);
                if (t != first+2)
                    first = t;
                break;
            case 'R':
                t = parse_binary_expression(first+2, last, "|=", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 'p':
            switch (t[1])
            {
            case 'm':
                t = parse_binary_expression(first+2, last, "->*", db);
                if (t != first+2)
                    first = t;
                break;
            case 'l':
                t = parse_binary_expression(first+2, last, "+", db);
                if (t != first+2)
                    first = t;
                break;
            case 'L':
                t = parse_binary_expression(first+2, last, "+=", db);
                if (t != first+2)
                    first = t;
                break;
            case 'p':
                if (first+2 != last && first[2] == '_')
                {
                    t = parse_prefix_expression(first+3, last, "++", db);
                    if (t != first+3)
                        first = t;
                }
                else
                {
                    const char* t1 = parse_expression(first+2, last, db);
                    if (t1 != first+2)
                    {
                        if (db.names.empty())
                            return first;
                        db.names.back() = "(" + db.names.back().move_full() + ")++";
                        first = t1;
                    }
                }
                break;
            case 's':
                t = parse_prefix_expression(first+2, last, "+", db);
                if (t != first+2)
                    first = t;
                break;
            case 't':
                first = parse_arrow_expr(first, last, db);
                break;
            }
            break;
        case 'q':
            if (t[1] == 'u')
            {
                const char* t1 = parse_expression(first+2, last, db);
                if (t1 != first+2)
                {
                    const char* t2 = parse_expression(t1, last, db);
                    if (t2 != t1)
                    {
                        const char* t3 = parse_expression(t2, last, db);
                        if (t3 != t2)
                        {
                            if (db.names.size() < 3)
                                return first;
                            auto op3 = db.names.back().move_full();
                            db.names.pop_back();
                            auto op2 = db.names.back().move_full();
                            db.names.pop_back();
                            auto op1 = db.names.back().move_full();
                            db.names.back() = "(" + op1 + ") ? (" + op2 + ") : (" + op3 + ")";
                            first = t3;
                        }
                        else
                        {
                            if (db.names.size() < 2)
                              return first;
                            db.names.pop_back();
                            db.names.pop_back();
                        }
                    }
                    else if (!db.names.empty())
                        db.names.pop_back();
                }
            }
            break;
        case 'r':
            switch (t[1])
            {
            case 'c':
                first = parse_reinterpret_cast_expr(first, last, db);
                break;
            case 'm':
                t = parse_binary_expression(first+2, last, "%", db);
                if (t != first+2)
                    first = t;
                break;
            case 'M':
                t = parse_binary_expression(first+2, last, "%=", db);
                if (t != first+2)
                    first = t;
                break;
            case 's':
                t = parse_binary_expression(first+2, last, ">>", db);
                if (t != first+2)
                    first = t;
                break;
            case 'S':
                t = parse_binary_expression(first+2, last, ">>=", db);
                if (t != first+2)
                    first = t;
                break;
            }
            break;
        case 's':
            switch (t[1])
            {
            case 'c':
                first = parse_static_cast_expr(first, last, db);
                break;
            case 'p':
                first = parse_pack_expansion(first, last, db);
                break;
            case 'r':
                return parse_unresolved_name(first, last, db);
            case 't':
                first = parse_sizeof_type_expr(first, last, db);
                break;
            case 'z':
                first = parse_sizeof_expr_expr(first, last, db);
                break;
            case 'Z':
                if (last - t >= 3)
                {
                    switch (t[2])
                    {
                    case 'T':
                        first = parse_sizeof_param_pack_expr(first, last, db);
                        break;
                    case 'f':
                        first = parse_sizeof_function_param_pack_expr(first, last, db);
                        break;
                    }
                }
                break;
            }
            break;
        case 't':
            switch (t[1])
            {
            case 'e':
            case 'i':
                first = parse_typeid_expr(first, last, db);
                break;
            case 'r':
                db.names.push_back("throw");
                first += 2;
                break;
            case 'w':
                first = parse_throw_expr(first, last, db);
                break;
            }
            break;
        case '1':
        case '2':
        case '3':
        case '4':
        case '5':
        case '6':
        case '7':
        case '8':
        case '9':
            return parse_unresolved_name(first, last, db);
        }
    }
    return first;
}

// <template-arg> ::= <type>                                             # type or template
//                ::= X <expression> E                                   # expression
//                ::= <expr-primary>                                     # simple expressions
//                ::= J <template-arg>* E                                # argument pack
//                ::= LZ <encoding> E                                    # extension

template <class C>
const char*
parse_template_arg(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        const char* t;
        switch (*first)
        {
        case 'X':
            t = parse_expression(first+1, last, db);
            if (t != first+1)
            {
                if (t != last && *t == 'E')
                    first = t+1;
            }
            break;
        case 'J':
            t = first+1;
            if (t == last)
                return first;
            while (*t != 'E')
            {
                const char* t1 = parse_template_arg(t, last, db);
                if (t1 == t)
                    return first;
                t = t1;
            }
            first = t+1;
            break;
        case 'L':
            // <expr-primary> or LZ <encoding> E
            if (first+1 != last && first[1] == 'Z')
            {
                t = parse_encoding(first+2, last, db);
                if (t != first+2 && t != last && *t == 'E')
                    first = t+1;
            }
            else
                first = parse_expr_primary(first, last, db);
            break;
        default:
            // <type>
            first = parse_type(first, last, db);
            break;
        }
    }
    return first;
}

// <template-args> ::= I <template-arg>* E
//     extension, the abi says <template-arg>+

template <class C>
const char*
parse_template_args(const char* first, const char* last, C& db)
{
    if (last - first >= 2 && *first == 'I')
    {
        if (db.tag_templates)
            db.template_param.back().clear();
        const char* t = first+1;
        typename C::String args("<");
        while (*t != 'E')
        {
            if (db.tag_templates)
                db.template_param.emplace_back(db.names.get_allocator());
            size_t k0 = db.names.size();
            const char* t1 = parse_template_arg(t, last, db);
            size_t k1 = db.names.size();
            if (db.tag_templates)
                db.template_param.pop_back();
            if (t1 == t || t1 == last)
                return first;
            if (db.tag_templates)
            {
                db.template_param.back().emplace_back(db.names.get_allocator());
                for (size_t k = k0; k < k1; ++k)
                    db.template_param.back().back().push_back(db.names[k]);
            }
            for (size_t k = k0; k < k1; ++k)
            {
                if (args.size() > 1)
                    args += ", ";
                args += db.names[k].move_full();
            }
            for (; k1 > k0; --k1)
                if (!db.names.empty())
                    db.names.pop_back();
            t = t1;
        }
        first = t + 1;
        if (args.back() != '>')
            args += ">";
        else
            args += " >";
        db.names.push_back(std::move(args));
        
    }
    return first;
}

// <nested-name> ::= N [<CV-qualifiers>] [<ref-qualifier>] <prefix> <unqualified-name> E
//               ::= N [<CV-qualifiers>] [<ref-qualifier>] <template-prefix> <template-args> E
// 
// <prefix> ::= <prefix> <unqualified-name>
//          ::= <template-prefix> <template-args>
//          ::= <template-param>
//          ::= <decltype>
//          ::= # empty
//          ::= <substitution>
//          ::= <prefix> <data-member-prefix>
//  extension ::= L
// 
// <template-prefix> ::= <prefix> <template unqualified-name>
//                   ::= <template-param>
//                   ::= <substitution>

template <class C>
const char*
parse_nested_name(const char* first, const char* last, C& db,
                  bool* ends_with_template_args)
{
    if (first != last && *first == 'N')
    {
        unsigned cv;
        const char* t0 = parse_cv_qualifiers(first+1, last, cv);
        if (t0 == last)
            return first;
        db.ref = 0;
        if (*t0 == 'R')
        {
            db.ref = 1;
            ++t0;
        }
        else if (*t0 == 'O')
        {
            db.ref = 2;
            ++t0;
        }
        db.names.emplace_back();
        if (last - t0 >= 2 && t0[0] == 'S' && t0[1] == 't')
        {
            t0 += 2;
            db.names.back().first = "std";
        }
        if (t0 == last)
        {
            db.names.pop_back();
            return first;
        }
        bool pop_subs = false;
        bool component_ends_with_template_args = false;
        while (*t0 != 'E')
        {
            component_ends_with_template_args = false;
            const char* t1;
            switch (*t0)
            {
            case 'S':
                if (t0 + 1 != last && t0[1] == 't')
                    goto do_parse_unqualified_name;
                t1 = parse_substitution(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    auto name = db.names.back().move_full();
                    db.names.pop_back();
                    if (db.names.empty())
                        return first;
                    if (!db.names.back().first.empty())
                    {
                        db.names.back().first += "::" + name;
                        db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                    }
                    else
                        db.names.back().first = name;
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
                break;
            case 'T':
                t1 = parse_template_param(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    auto name = db.names.back().move_full();
                    db.names.pop_back();
                    if (db.names.empty())
                        return first;
                    if (!db.names.back().first.empty())
                        db.names.back().first += "::" + name;
                    else
                        db.names.back().first = name;
                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
                break;
            case 'D':
                if (t0 + 1 != last && t0[1] != 't' && t0[1] != 'T')
                    goto do_parse_unqualified_name;
                t1 = parse_decltype(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    auto name = db.names.back().move_full();
                    db.names.pop_back();
                    if (db.names.empty())
                        return first;
                    if (!db.names.back().first.empty())
                        db.names.back().first += "::" + name;
                    else
                        db.names.back().first = name;
                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
                break;
            case 'I':
                t1 = parse_template_args(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    auto name = db.names.back().move_full();
                    db.names.pop_back();
                    if (db.names.empty())
                        return first;
                    db.names.back().first += name;
                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                    t0 = t1;
                    component_ends_with_template_args = true;
                }
                else
                    return first;
                break;
            case 'L':
                if (++t0 == last)
                    return first;
                break;
            default:
            do_parse_unqualified_name:
                t1 = parse_unqualified_name(t0, last, db);
                if (t1 != t0 && t1 != last)
                {
                    auto name = db.names.back().move_full();
                    db.names.pop_back();
                    if (db.names.empty())
                        return first;
                    if (!db.names.back().first.empty())
                        db.names.back().first += "::" + name;
                    else
                        db.names.back().first = name;
                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                    pop_subs = true;
                    t0 = t1;
                }
                else
                    return first;
            }
        }
        first = t0 + 1;
        db.cv = cv;
        if (pop_subs && !db.subs.empty())
            db.subs.pop_back();
        if (ends_with_template_args)
            *ends_with_template_args = component_ends_with_template_args;
    }
    return first;
}

// <discriminator> := _ <non-negative number>      # when number < 10
//                 := __ <non-negative number> _   # when number >= 10
//  extension      := decimal-digit+               # at the end of string

const char*
parse_discriminator(const char* first, const char* last)
{
    // parse but ignore discriminator
    if (first != last)
    {
        if (*first == '_')
        {
            const char* t1 = first+1;
            if (t1 != last)
            {
                if (std::isdigit(*t1))
                    first = t1+1;
                else if (*t1 == '_')
                {
                    for (++t1; t1 != last && std::isdigit(*t1); ++t1)
                        ;
                    if (t1 != last && *t1 == '_')
                        first = t1 + 1;
                }
            }
        }
        else if (std::isdigit(*first))
        {
            const char* t1 = first+1;
            for (; t1 != last && std::isdigit(*t1); ++t1)
                ;
            if (t1 == last)
                first = last;
        }
    }
    return first;
}

// <local-name> := Z <function encoding> E <entity name> [<discriminator>]
//              := Z <function encoding> E s [<discriminator>]
//              := Z <function encoding> Ed [ <parameter number> ] _ <entity name>

template <class C>
const char*
parse_local_name(const char* first, const char* last, C& db,
                 bool* ends_with_template_args)
{
    if (first != last && *first == 'Z')
    {
        const char* t = parse_encoding(first+1, last, db);
        if (t != first+1 && t != last && *t == 'E' && ++t != last)
        {
            switch (*t)
            {
            case 's':
                first = parse_discriminator(t+1, last);
                if (db.names.empty())
                    return first;
                db.names.back().first.append("::string literal");
                break;
            case 'd':
                if (++t != last)
                {
                    const char* t1 = parse_number(t, last);
                    if (t1 != last && *t1 == '_')
                    {
                        t = t1 + 1;
                        t1 = parse_name(t, last, db,
                                        ends_with_template_args);
                        if (t1 != t)
                        {
                            if (db.names.size() < 2)
                                return first;
                            auto name = db.names.back().move_full();
                            db.names.pop_back();
                            if (db.names.empty())
                                return first;
                            db.names.back().first.append("::");
                            db.names.back().first.append(name);
                            first = t1;
                        }
                        else if (!db.names.empty())
                            db.names.pop_back();
                    }
                }
                break;
            default:
                {
                    const char* t1 = parse_name(t, last, db,
                                                ends_with_template_args);
                    if (t1 != t)
                    {
                        // parse but ignore discriminator
                        first = parse_discriminator(t1, last);
                        if (db.names.size() < 2)
                            return first;
                        auto name = db.names.back().move_full();
                        db.names.pop_back();
                        if (db.names.empty())
                            return first;
                        db.names.back().first.append("::");
                        db.names.back().first.append(name);
                    }
                    else if (!db.names.empty())
                        db.names.pop_back();
                }
                break;
            }
        }
    }
    return first;
}

// <name> ::= <nested-name> // N
//        ::= <local-name> # See Scope Encoding below  // Z
//        ::= <unscoped-template-name> <template-args>
//        ::= <unscoped-name>

// <unscoped-template-name> ::= <unscoped-name>
//                          ::= <substitution>

template <class C>
const char*
parse_name(const char* first, const char* last, C& db,
           bool* ends_with_template_args)
{
    if (last - first >= 2)
    {
        const char* t0 = first;
        // extension: ignore L here
        if (*t0 == 'L')
            ++t0;
        switch (*t0)
        {
        case 'N':
          {
            const char* t1 = parse_nested_name(t0, last, db,
                                               ends_with_template_args);
            if (t1 != t0)
                first = t1;
            break;
          }
        case 'Z':
          {
            const char* t1 = parse_local_name(t0, last, db,
                                              ends_with_template_args);
            if (t1 != t0)
                first = t1;
            break;
          }
        default:
          {
            const char* t1 = parse_unscoped_name(t0, last, db);
            if (t1 != t0)
            {
                if (t1 != last && *t1 == 'I')  // <unscoped-template-name> <template-args>
                {
                    if (db.names.empty())
                        return first;
                    db.subs.push_back(typename C::sub_type(1, db.names.back(), db.names.get_allocator()));
                    t0 = t1;
                    t1 = parse_template_args(t0, last, db);
                    if (t1 != t0)
                    {
                        if (db.names.size() < 2)
                            return first;
                        auto tmp = db.names.back().move_full();
                        db.names.pop_back();
                        if (db.names.empty())
                            return first;
                        db.names.back().first += tmp;
                        first = t1;
                        if (ends_with_template_args)
                            *ends_with_template_args = true;
                    }
                }
                else   // <unscoped-name>
                    first = t1;
            }
            else
            {   // try <substitution> <template-args>
                t1 = parse_substitution(t0, last, db);
                if (t1 != t0 && t1 != last && *t1 == 'I')
                {
                    t0 = t1;
                    t1 = parse_template_args(t0, last, db);
                    if (t1 != t0)
                    {
                        if (db.names.size() < 2)
                            return first;
                        auto tmp = db.names.back().move_full();
                        db.names.pop_back();
                        if (db.names.empty())
                            return first;
                        db.names.back().first += tmp;
                        first = t1;
                        if (ends_with_template_args)
                            *ends_with_template_args = true;
                    }
                }
            }
            break;
          }
        }
    }
    return first;
}

// <call-offset> ::= h <nv-offset> _
//               ::= v <v-offset> _
// 
// <nv-offset> ::= <offset number>
//               # non-virtual base override
// 
// <v-offset>  ::= <offset number> _ <virtual offset number>
//               # virtual base override, with vcall offset

const char*
parse_call_offset(const char* first, const char* last)
{
    if (first != last)
    {
        switch (*first)
        {
        case 'h':
            {
            const char* t = parse_number(first + 1, last);
            if (t != first + 1 && t != last && *t == '_')
                first = t + 1;
            }
            break;
        case 'v':
            {
            const char* t = parse_number(first + 1, last);
            if (t != first + 1 && t != last && *t == '_')
            {
                const char* t2 = parse_number(++t, last);
                if (t2 != t && t2 != last && *t2 == '_')
                    first = t2 + 1;
            }
            }
            break;
        }
    }
    return first;
}

// <special-name> ::= TV <type>    # virtual table
//                ::= TT <type>    # VTT structure (construction vtable index)
//                ::= TI <type>    # typeinfo structure
//                ::= TS <type>    # typeinfo name (null-terminated byte string)
//                ::= Tc <call-offset> <call-offset> <base encoding>
//                    # base is the nominal target function of thunk
//                    # first call-offset is 'this' adjustment
//                    # second call-offset is result adjustment
//                ::= T <call-offset> <base encoding>
//                    # base is the nominal target function of thunk
//                ::= GV <object name> # Guard variable for one-time initialization
//                                     # No <type>
//                ::= TW <object name> # Thread-local wrapper
//                ::= TH <object name> # Thread-local initialization
//      extension ::= TC <first type> <number> _ <second type> # construction vtable for second-in-first
//      extension ::= GR <object name> # reference temporary for object

template <class C>
const char*
parse_special_name(const char* first, const char* last, C& db)
{
    if (last - first > 2)
    {
        const char* t;
        switch (*first)
        {
        case 'T':
            switch (first[1])
            {
            case 'V':
                // TV <type>    # virtual table
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "vtable for ");
                    first = t;
                }
                break;
            case 'T':
                // TT <type>    # VTT structure (construction vtable index)
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "VTT for ");
                    first = t;
                }
                break;
            case 'I':
                // TI <type>    # typeinfo structure
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "typeinfo for ");
                    first = t;
                }
                break;
            case 'S':
                // TS <type>    # typeinfo name (null-terminated byte string)
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "typeinfo name for ");
                    first = t;
                }
                break;
            case 'c':
                // Tc <call-offset> <call-offset> <base encoding>
              {
                const char* t0 = parse_call_offset(first+2, last);
                if (t0 == first+2)
                    break;
                const char* t1 = parse_call_offset(t0, last);
                if (t1 == t0)
                    break;
                t = parse_encoding(t1, last, db);
                if (t != t1)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "covariant return thunk to ");
                    first = t;
                }
              }
                break;
            case 'C':
                // extension ::= TC <first type> <number> _ <second type> # construction vtable for second-in-first
                t = parse_type(first+2, last, db);
                if (t != first+2)
                {
                    const char* t0 = parse_number(t, last);
                    if (t0 != t && t0 != last && *t0 == '_')
                    {
                        const char* t1 = parse_type(++t0, last, db);
                        if (t1 != t0)
                        {
                            if (db.names.size() < 2)
                                return first;
                            auto left = db.names.back().move_full();
                            db.names.pop_back();
                            if (db.names.empty())
                                return first;
                            db.names.back().first = "construction vtable for " +
                                                    std::move(left) + "-in-" +
                                                    db.names.back().move_full();
                            first = t1;
                        }
                    }
                }
                break;
            case 'W':
                // TW <object name> # Thread-local wrapper
                t = parse_name(first + 2, last, db);
                if (t != first + 2) 
                {
                    if (db.names.empty())
                    return first;
                    db.names.back().first.insert(0, "thread-local wrapper routine for ");
                    first = t;
                }
                break;
            case 'H':
                //TH <object name> # Thread-local initialization
                t = parse_name(first + 2, last, db);
                if (t != first + 2) 
                {
                    if (db.names.empty())
                    return first;
                    db.names.back().first.insert(0, "thread-local initialization routine for ");
                    first = t;
                }
                break;
            default:
                // T <call-offset> <base encoding>
                {
                const char* t0 = parse_call_offset(first+1, last);
                if (t0 == first+1)
                    break;
                t = parse_encoding(t0, last, db);
                if (t != t0)
                {
                    if (db.names.empty())
                        return first;
                    if (first[1] == 'v')
                    {
                        db.names.back().first.insert(0, "virtual thunk to ");
                        first = t;
                    }
                    else
                    {
                        db.names.back().first.insert(0, "non-virtual thunk to ");
                        first = t;
                    }
                }
                }
                break;
            }
            break;
        case 'G':
            switch (first[1])
            {
            case 'V':
                // GV <object name> # Guard variable for one-time initialization
                t = parse_name(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "guard variable for ");
                    first = t;
                }
                break;
            case 'R':
                // extension ::= GR <object name> # reference temporary for object
                t = parse_name(first+2, last, db);
                if (t != first+2)
                {
                    if (db.names.empty())
                        return first;
                    db.names.back().first.insert(0, "reference temporary for ");
                    first = t;
                }
                break;
            }
            break;
        }
    }
    return first;
}

template <class T>
class save_value
{
    T& restore_;
    T original_value_;
public:
    save_value(T& restore)
        : restore_(restore),
          original_value_(restore)
        {}

    ~save_value()
    {
        restore_ = std::move(original_value_);
    }

    save_value(const save_value&) = delete;
    save_value& operator=(const save_value&) = delete;
};

// <encoding> ::= <function name> <bare-function-type>
//            ::= <data name>
//            ::= <special-name>

template <class C>
const char*
parse_encoding(const char* first, const char* last, C& db)
{
    if (first != last)
    {
        save_value<decltype(db.encoding_depth)> su(db.encoding_depth);
        ++db.encoding_depth;
        save_value<decltype(db.tag_templates)> sb(db.tag_templates);
        if (db.encoding_depth > 1)
            db.tag_templates = true;
        switch (*first)
        {
        case 'G':
        case 'T':
            first = parse_special_name(first, last, db);
            break;
        default:
          {
            bool ends_with_template_args = false;
            const char* t = parse_name(first, last, db,
                                       &ends_with_template_args);
            unsigned cv = db.cv;
            unsigned ref = db.ref;
            if (t != first)
            {
                if (t != last && *t != 'E' && *t != '.')
                {
                    save_value<bool> sb2(db.tag_templates);
                    db.tag_templates = false;
                    const char* t2;
                    typename C::String ret2;
                    if (db.names.empty())
                        return first;
                    const typename C::String& nm = db.names.back().first;
                    if (nm.empty())
                        return first;
                    if (!db.parsed_ctor_dtor_cv && ends_with_template_args)
                    {
                        t2 = parse_type(t, last, db);
                        if (t2 == t)
                            return first;
                        if (db.names.size() < 2)
                            return first;
                        auto ret1 = std::move(db.names.back().first);
                        ret2 = std::move(db.names.back().second);
                        if (ret2.empty())
                            ret1 += ' ';
                        db.names.pop_back();
                        if (db.names.empty())
                            return first;

                        db.names.back().first.insert(0, ret1);
                        t = t2;
                    }
                    db.names.back().first += '(';
                    if (t != last && *t == 'v')
                    {
                        ++t;
                    }
                    else
                    {
                        bool first_arg = true;
                        while (true)
                        {
                            size_t k0 = db.names.size();
                            t2 = parse_type(t, last, db);
                            size_t k1 = db.names.size();
                            if (t2 == t)
                                break;
                            if (k1 > k0)
                            {
                                typename C::String tmp;
                                for (size_t k = k0; k < k1; ++k)
                                {
                                    if (!tmp.empty())
                                        tmp += ", ";
                                    tmp += db.names[k].move_full();
                                }
                                for (size_t k = k0; k < k1; ++k) {
                                    if (db.names.empty())
                                        return first;
                                    db.names.pop_back();
                                }
                                if (!tmp.empty())
                                {
                                    if (db.names.empty())
                                        return first;
                                    if (!first_arg)
                                        db.names.back().first += ", ";
                                    else
                                        first_arg = false;
                                    db.names.back().first += tmp;
                                }
                            }
                            t = t2;
                        }
                    }
                    if (db.names.empty())
                        return first;
                    db.names.back().first += ')';
                    if (cv & 1)
                        db.names.back().first.append(" const");
                    if (cv & 2)
                        db.names.back().first.append(" volatile");
                    if (cv & 4)
                        db.names.back().first.append(" restrict");
                    if (ref == 1)
                        db.names.back().first.append(" &");
                    else if (ref == 2)
                        db.names.back().first.append(" &&");
                    db.names.back().first += ret2;
                    first = t;
                }
                else
                    first = t;
            }
            break;
          }
        }
    }
    return first;
}

// _block_invoke
// _block_invoke<decimal-digit>+
// _block_invoke_<decimal-digit>+

template <class C>
const char*
parse_block_invoke(const char* first, const char* last, C& db)
{
    if (last - first >= 13)
    {
        const char test[] = "_block_invoke";
        const char* t = first;
        for (int i = 0; i < 13; ++i, ++t)
        {
            if (*t != test[i])
                return first;
        }
        if (t != last)
        {
            if (*t == '_')
            {
                // must have at least 1 decimal digit
                if (++t == last || !std::isdigit(*t))
                    return first;
                ++t;
            }
            // parse zero or more digits
            while (t != last && isdigit(*t))
                ++t;
        }
        if (db.names.empty())
            return first;
        db.names.back().first.insert(0, "invocation function for block in ");
        first = t;
    }
    return first;
}

// extension
// <dot-suffix> := .<anything and everything>

template <class C>
const char*
parse_dot_suffix(const char* first, const char* last, C& db)
{
    if (first != last && *first == '.')
    {
        if (db.names.empty())
            return first;
        db.names.back().first += " (" + typename C::String(first, last) + ")";
        first = last;
    }
    return first;
}

// <block-involcaton-function> ___Z<encoding>_block_invoke
// <block-involcaton-function> ___Z<encoding>_block_invoke<decimal-digit>+
// <block-involcaton-function> ___Z<encoding>_block_invoke_<decimal-digit>+
// <mangled-name> ::= _Z<encoding>
//                ::= <type>

template <class C>
void
demangle(const char* first, const char* last, C& db, int& status)
{
    if (first >= last)
    {
        status = invalid_mangled_name;
        return;
    }
    if (*first == '_')
    {
        if (last - first >= 4)
        {
            if (first[1] == 'Z')
            {
                const char* t = parse_encoding(first+2, last, db);
                if (t != first+2 && t != last && *t == '.')
                    t = parse_dot_suffix(t, last, db);
                if (t != last)
                    status = invalid_mangled_name;
            }
            else if (first[1] == '_' && first[2] == '_' && first[3] == 'Z')
            {
                const char* t = parse_encoding(first+4, last, db);
                if (t != first+4 && t != last)
                {
                    const char* t1 = parse_block_invoke(t, last, db);
                    if (t1 != last)
                        status = invalid_mangled_name;
                }
                else
                    status = invalid_mangled_name;
            }
            else
                status = invalid_mangled_name;
        }
        else
            status = invalid_mangled_name;
    }
    else
    {
        const char* t = parse_type(first, last, db);
        if (t != last)
            status = invalid_mangled_name;
    }
    if (status == success && db.names.empty())
        status = invalid_mangled_name;
}

template <std::size_t N>
class arena
{
    static const std::size_t alignment = 16;
    alignas(alignment) char buf_[N];
    char* ptr_;

    std::size_t 
    align_up(std::size_t n) noexcept
        {return (n + (alignment-1)) & ~(alignment-1);}

    bool
    pointer_in_buffer(char* p) noexcept
        {return buf_ <= p && p <= buf_ + N;}

public:
    arena() noexcept : ptr_(buf_) {}
    ~arena() {ptr_ = nullptr;}
    arena(const arena&) = delete;
    arena& operator=(const arena&) = delete;

    char* allocate(std::size_t n);
    void deallocate(char* p, std::size_t n) noexcept;

    static constexpr std::size_t size() {return N;}
    std::size_t used() const {return static_cast<std::size_t>(ptr_ - buf_);}
    void reset() {ptr_ = buf_;}
};

template <std::size_t N>
char*
arena<N>::allocate(std::size_t n)
{
    n = align_up(n);
    if (static_cast<std::size_t>(buf_ + N - ptr_) >= n)
    {
        char* r = ptr_;
        ptr_ += n;
        return r;
    }
    return static_cast<char*>(std::malloc(n));
}

template <std::size_t N>
void
arena<N>::deallocate(char* p, std::size_t n) noexcept
{
    if (pointer_in_buffer(p))
    {
        n = align_up(n);
        if (p + n == ptr_)
            ptr_ = p;
    }
    else
        std::free(p);
}

template <class T, std::size_t N>
class short_alloc
{
    arena<N>& a_;
public:
    typedef T value_type;

public:
    template <class _Up> struct rebind {typedef short_alloc<_Up, N> other;};

    short_alloc(arena<N>& a) noexcept : a_(a) {}
    template <class U>
        short_alloc(const short_alloc<U, N>& a) noexcept
            : a_(a.a_) {}
    short_alloc(const short_alloc&) = default;
    short_alloc& operator=(const short_alloc&) = delete;

    T* allocate(std::size_t n)
    {
        return reinterpret_cast<T*>(a_.allocate(n*sizeof(T)));
    }
    void deallocate(T* p, std::size_t n) noexcept
    {
        a_.deallocate(reinterpret_cast<char*>(p), n*sizeof(T));
    }

    template <class T1, std::size_t N1, class U, std::size_t M>
    friend
    bool
    operator==(const short_alloc<T1, N1>& x, const short_alloc<U, M>& y) noexcept;

    template <class U, std::size_t M> friend class short_alloc;
};

template <class T, std::size_t N, class U, std::size_t M>
inline
bool
operator==(const short_alloc<T, N>& x, const short_alloc<U, M>& y) noexcept
{
    return N == M && &x.a_ == &y.a_;
}

template <class T, std::size_t N, class U, std::size_t M>
inline
bool
operator!=(const short_alloc<T, N>& x, const short_alloc<U, M>& y) noexcept
{
    return !(x == y);
}

template <class T>
class malloc_alloc
{
public:
    typedef T value_type;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T* pointer;
    typedef const T* const_pointer;
    typedef std::size_t size_type;
    typedef std::ptrdiff_t difference_type;

    malloc_alloc() = default;
    template <class U> malloc_alloc(const malloc_alloc<U>&) noexcept {}

    T* allocate(std::size_t n)
    {
        return static_cast<T*>(std::malloc(n*sizeof(T)));
    }
    void deallocate(T* p, std::size_t) noexcept
    {
        std::free(p);
    }

    template <class U> struct rebind { using other = malloc_alloc<U>; };
    template <class U, class... Args>
    void construct(U* p, Args&&... args)
    {
        ::new ((void*)p) U(std::forward<Args>(args)...);
    }
    void destroy(T* p)
    {
        p->~T();
    }
};

template <class T, class U>
inline
bool
operator==(const malloc_alloc<T>&, const malloc_alloc<U>&) noexcept
{
    return true;
}

template <class T, class U>
inline
bool
operator!=(const malloc_alloc<T>& x, const malloc_alloc<U>& y) noexcept
{
    return !(x == y);
}

const size_t bs = 4 * 1024;
template <class T> using Alloc = short_alloc<T, bs>;
template <class T> using Vector = std::vector<T, Alloc<T>>;

template <class StrT>
struct string_pair
{
    StrT first;
    StrT second;

    string_pair() = default;
    string_pair(StrT f) : first(std::move(f)) {}
    string_pair(StrT f, StrT s)
        : first(std::move(f)), second(std::move(s)) {}
    template <size_t N>
        string_pair(const char (&s)[N]) : first(s, N-1) {}

    size_t size() const {return first.size() + second.size();}
    StrT full() const {return first + second;}
    StrT move_full() {return std::move(first) + std::move(second);}
};

struct Db
{
    typedef std::basic_string<char, std::char_traits<char>,
                              malloc_alloc<char>> String;
    typedef Vector<string_pair<String>> sub_type;
    typedef Vector<sub_type> template_param_type;
    sub_type names;
    template_param_type subs;
    Vector<template_param_type> template_param;
    unsigned cv = 0;
    unsigned ref = 0;
    unsigned encoding_depth = 0;
    bool parsed_ctor_dtor_cv = false;
    bool tag_templates = true;
    bool fix_forward_references = false;
    bool try_to_parse_template_args = true;

    template <size_t N>
    Db(arena<N>& ar) :
        names(ar),
        subs(0, names, ar),
        template_param(0, subs, ar)
    {}
};

}  // unnamed namespace

extern "C" _LIBCXXABI_FUNC_VIS char *
__cxa_demangle(const char *mangled_name, char *buf, size_t *n, int *status) {
    if (mangled_name == nullptr || (buf != nullptr && n == nullptr))
    {
        if (status)
            *status = invalid_args;
        return nullptr;
    }

    size_t internal_size = buf != nullptr ? *n : 0;
    arena<bs> a;
    Db db(a);
    db.template_param.emplace_back(a);
    int internal_status = success;
    size_t len = std::strlen(mangled_name);
    demangle(mangled_name, mangled_name + len, db,
             internal_status);
    if (internal_status == success && db.fix_forward_references &&
           !db.template_param.empty() && !db.template_param.front().empty())
    {
        db.fix_forward_references = false;
        db.tag_templates = false;
        db.names.clear();
        db.subs.clear();
        demangle(mangled_name, mangled_name + len, db, internal_status);
        if (db.fix_forward_references)
            internal_status = invalid_mangled_name;
    }
    if (internal_status == success)
    {
        size_t sz = db.names.back().size() + 1;
        if (sz > internal_size)
        {
            char* newbuf = static_cast<char*>(std::realloc(buf, sz));
            if (newbuf == nullptr)
            {
                internal_status = memory_alloc_failure;
                buf = nullptr;
            }
            else
            {
                buf = newbuf;
                if (n != nullptr)
                    *n = sz;
            }
        }
        if (buf != nullptr)
        {
            db.names.back().first += db.names.back().second;
            std::memcpy(buf, db.names.back().first.data(), sz-1);
            buf[sz-1] = char(0);
        }
    }
    else
        buf = nullptr;
    if (status)
        *status = internal_status;
    return buf;
}

}  // __cxxabiv1
