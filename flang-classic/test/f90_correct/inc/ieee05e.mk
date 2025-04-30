#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee05e  ########

CWD   := $(shell pwd)
INVOKEE=runieee

ieee05e: ieee05e.$(OBJX)

ieee05e.$(OBJX):  $(SRC)/ieee05e.f90
	-$(RM) ieee05e.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	@echo $(CWD)/ieee05e.$(EXESUFFIX) > $(INVOKEE)
	chmod 744 $(INVOKEE)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee05e.f90 -o ieee05e.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee05e.$(OBJX) check.$(OBJX) $(LIBS) -o ieee05e.$(EXESUFFIX)


ieee05e.run: ieee05e.$(OBJX)
	@echo ------------------------------------ executing test ieee05e
	$(shell ./$(INVOKEE) > ieee05e.res 2> ieee05e.err)
	@cat ieee05e.res
run: ieee05e.$(OBJX)
	@echo ------------------------------------ executing test ieee05e
	$(shell ./$(INVOKEE) > ieee05e.res 2> ieee05e.err)
	@cat ieee05e.res
build:	ieee05e.$(OBJX)
verify:	;

