#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop400  ########


oop400: run
	

build: clean $(SRC)/oop400.f90
	-$(RM) oop400.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop400.f90 -o oop400.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop400.$(OBJX) check.$(OBJX) $(LIBS) -o oop400.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop400
	oop400.$(EXESUFFIX)

verify: ;

