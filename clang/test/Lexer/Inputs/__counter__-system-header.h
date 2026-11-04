#define COUNTER_ALIAS __COUNTER__
#define COUNTER_MACRO() __COUNTER__

int header_counter_value = __COUNTER__;
int header_counter_alias = COUNTER_ALIAS;
int header_counter_macro = COUNTER_MACRO();

