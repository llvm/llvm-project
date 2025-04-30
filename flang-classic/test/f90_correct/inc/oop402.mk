#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop402  ########


oop402: run
	

build:  $(SRC)/oop402.f90
	-$(RM) oop402.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop402.f90 -o oop402.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop402.$(OBJX) check.$(OBJX) $(LIBS) -o oop402.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop402
	oop402.$(EXESUFFIX)

verify: ;

