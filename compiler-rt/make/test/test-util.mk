include make/util.mk

streq_t0 = $(call streq,,)
$(call AssertEqual,streq_t0,true)
streq_t1 = $(call streq,b,)
$(call AssertEqual,streq_t1,)
streq_t2 = $(call streq,,b)
$(call AssertEqual,streq_t2,)
streq_t3 = $(call streq,b,b)
$(call AssertEqual,streq_t3,true)
streq_t4 = $(call streq,bb,b)
$(call AssertEqual,streq_t4,)
streq_t5 = $(call streq,b,bb)
$(call AssertEqual,streq_t5,)
streq_t6 = $(call streq,bb,bb)
$(call AssertEqual,streq_t6,true)

strneq_t7 = $(call strneq,,)
$(call AssertEqual,strneq_t7,)
strneq_t8 = $(call strneq,b,)
$(call AssertEqual,strneq_t8,true)
strneq_t9 = $(call strneq,,b)
$(call AssertEqual,strneq_t9,true)
strneq_t10 = $(call strneq,b,b)
$(call AssertEqual,strneq_t10,)
strneq_t11 = $(call strneq,bb,b)
$(call AssertEqual,strneq_t11,true)
strneq_t12 = $(call strneq,b,bb)
$(call AssertEqual,strneq_t12,true)
strneq_t13 = $(call strneq,bb,bb)
$(call AssertEqual,strneq_t13,)

contains_t0 = $(call contains,a b b c,a)
$(call AssertEqual,contains_t0,true)
contains_t1 = $(call contains,a b b c,b)
$(call AssertEqual,contains_t1,true)
contains_t2 = $(call contains,a b b c,c)
$(call AssertEqual,contains_t2,true)
contains_t3 = $(call contains,a b b c,d)
$(call AssertEqual,contains_t3,)

isdefined_t0_defined_var := 0
isdefined_t0 = $(call IsDefined,isdefined_t0_defined_var)
$(call AssertEqual,isdefined_t0,true)
isdefined_t1 = $(call IsDefined,isdefined_t1_never_defined_var)
$(call AssertEqual,isdefined_t1,)

varordefault_t0_var := 1
varordefault_t0 = $(call VarOrDefault,varordefault_t0_var.opt,$(varordefault_t0_var))
$(call AssertEqual,varordefault_t0,1)
varordefault_t1_var := 1
varordefault_t1_var.opt := 2
varordefault_t1 = $(call VarOrDefault,varordefault_t1_var.opt,$(varordefault_t1_var))
$(call AssertEqual,varordefault_t1,2)

$(call CopyVariable,copyvariable_t0_src,copyvariable_t0_dst)
copyvariable_t0 = $(call IsUndefined,copyvariable_t0_dst)
$(call AssertEqual,copyvariable_t0,true)
copyvariable_t1_src = 1
$(call CopyVariable,copyvariable_t1_src,copyvariable_t1)
$(call AssertEqual,copyvariable_t1,1)

all:
	@true
.PHONY: all

