#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee05b  ########

CWD   := $(shell pwd)
INVOKEE=runieee

ieee05b: ieee05b.$(OBJX)
	

ieee05b.$(OBJX):  $(SRC)/ieee05b.f90
	-$(RM) ieee05b.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	@echo $(CWD)/ieee05b.$(EXESUFFIX) > $(INVOKEE)
	chmod 744 $(INVOKEE)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee05b.f90 -o ieee05b.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee05b.$(OBJX) check.$(OBJX) $(LIBS) -o ieee05b.$(EXESUFFIX)


ieee05b.run: ieee05b.$(OBJX)
	@echo ------------------------------------ executing test ieee05b
	$(shell ./$(INVOKEE) > ieee05b.res 2> ieee05b.err)
	@cat ieee05b.res

run: ieee05b.$(OBJX)
	@echo ------------------------------------ executing test ieee05b
	$(shell ./$(INVOKEE) > ieee05b.res 2> ieee05b.err)
	@cat ieee05b.res
build:	ieee05b.$(OBJX)
verify:	;
