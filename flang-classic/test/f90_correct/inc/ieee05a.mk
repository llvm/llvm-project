#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee05a  ########

CWD   := $(shell pwd)
INVOKEE=runieee

ieee05a: ieee05a.$(OBJX)

ieee05a.$(OBJX):  $(SRC)/ieee05a.f90
	-$(RM) ieee05a.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	@echo $(CWD)/ieee05a.$(EXESUFFIX) > $(INVOKEE)
	chmod 744 $(INVOKEE)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee05a.f90 -o ieee05a.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee05a.$(OBJX) check.$(OBJX) $(LIBS) -o ieee05a.$(EXESUFFIX)


ieee05a.run: ieee05a.$(OBJX)
	@echo ------------------------------------ executing test ieee05a
	$(shell ./$(INVOKEE) > ieee05a.res 2> ieee05a.err)
	@cat ieee05a.res

run: ieee05a.$(OBJX)
	@echo ------------------------------------ executing test ieee05a
	$(shell ./$(INVOKEE) > ieee05a.res 2> ieee05a.err)
	@cat ieee05a.res

build:	ieee05a.$(OBJX)
verify:	;
