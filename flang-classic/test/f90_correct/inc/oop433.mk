#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop433  ########


oop433: run
	

build:  $(SRC)/oop433.f90
	-$(RM) oop433.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop433.f90 -o oop433.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop433.$(OBJX) check.$(OBJX) $(LIBS) -o oop433.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop433
	oop433.$(EXESUFFIX)

verify: ;

