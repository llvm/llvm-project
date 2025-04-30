#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ieee05d  ########

CWD   := $(shell pwd)
INVOKEE=runieee

ieee05d: ieee05d.$(OBJX)
	
ieee05d.$(OBJX):  $(SRC)/ieee05d.f90
	-$(RM) ieee05d.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	@echo $(CWD)/ieee05d.$(EXESUFFIX) > $(INVOKEE)
	chmod 744 $(INVOKEE)
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ieee05d.f90 -o ieee05d.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ieee05d.$(OBJX) check.$(OBJX) $(LIBS) -o ieee05d.$(EXESUFFIX)


ieee05d.run: ieee05d.$(OBJX)
	@echo ------------------------------------ executing test ieee05d
	$(shell ./$(INVOKEE) > ieee05d.res 2> ieee05d.err)
	@cat ieee05d.res

run: ieee05d.$(OBJX)
	@echo ------------------------------------ executing test ieee05d
	$(shell ./$(INVOKEE) > ieee05d.res 2> ieee05d.err)
	@cat ieee05d.res
build:	ieee05d.$(OBJX)
verify:	;
