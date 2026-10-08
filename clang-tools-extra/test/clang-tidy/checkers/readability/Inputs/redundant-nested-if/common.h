#ifndef READABILITY_REDUNDANT_NESTED_IF_COMMON_H
#define READABILITY_REDUNDANT_NESTED_IF_COMMON_H

bool cond(int X = 0);
int side_effect();
void sink();
void bar();

struct BoolLike {
  operator bool() const;
};

BoolLike make_bool_like();

#define INNER_IF(C) if (C) sink()
#define COND_MACRO cond()
#define OUTER_IF if (cond())

#endif // READABILITY_REDUNDANT_NESTED_IF_COMMON_H
