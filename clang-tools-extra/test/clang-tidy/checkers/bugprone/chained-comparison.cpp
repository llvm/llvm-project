// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-chained-comparison %t

void badly_chained_1(int x, int y, int z)
{
    bool result = x < y < z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 < v1 < v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_2(int x, int y, int z)
{
    bool result = x <= y <= z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 <= v1 <= v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_3(int x, int y, int z)
{
    bool result = x > y > z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 > v1 > v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_4(int x, int y, int z)
{
    bool result = x >= y >= z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 >= v1 >= v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_5(int x, int y, int z)
{
    bool result = x == y != z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 == v1 != v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_6(bool x, bool y, bool z)
{
    bool result = x != y == z;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 != v1 == v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_multiple(bool a, bool b, bool c, bool d, bool e, bool f, bool g, bool h)
{
    bool result = a == b == c == d == e == f == g == h;
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 == v1 == v2 == v3 == v4 == v5 == v6 == v7' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_limit(bool v[29])
{
// CHECK-MESSAGES: :[[@LINE+1]]:19: warning: chained comparison 'v0 == v1 == v2 == v3 == v4 == v5 == v6 == v7 == v8 == v9 == v10 == v11 == v12 == v13 == v14 == v15 == v16 == v17 == v18 == v19 == v20 == v21 == v22 == v23 == v24 == v25 == v26 == v27 == v28' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]
    bool result = v[0] == v[1] == v[2] == v[3] == v[4] == v[5] == v[6] == v[7] ==
                  v[8] == v[9] == v[10] == v[11] == v[12] == v[13] == v[14] ==
                  v[15] == v[16] == v[17] == v[18] == v[19] == v[20] == v[21] ==
                  v[22] == v[23] == v[24] == v[25] == v[26] == v[27] == v[28];

}

void badly_chained_parens2(int x, int y, int z, int t, int a, int b)
{
    bool result = (x < y) < (z && t) > (a == b);
}
// CHECK-MESSAGES: :[[@LINE-2]]:19: warning: chained comparison 'v0 < v1 > v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void badly_chained_inner(int x, int y, int z, int t, int u)
{
    bool result = x && y < z < t && u;
}
// CHECK-MESSAGES: :[[@LINE-2]]:24: warning: chained comparison 'v0 < v1 < v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

void properly_chained_1(int x, int y, int z)
{
    bool result = x < y && y < z;
}

void properly_chained_2(int x, int y, bool z)
{
    bool result = (x < y) == z;
}

struct Value {
    bool operator<(const Value&) const;
};

bool operator==(bool, const Value&);

bool badWithCppOperator(Value a, Value b, Value c) {
    return a < b == c;
}
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: chained comparison 'v0 < v1 == v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]

bool mixedBinaryAndCpp(Value a, Value b, bool c) {
    return a < b == c;
}
// CHECK-MESSAGES: :[[@LINE-2]]:12: warning: chained comparison 'v0 < v1 == v2' may generate unintended results, use parentheses to specify order of evaluation or a logical operator to separate comparison expressions [bugprone-chained-comparison]
