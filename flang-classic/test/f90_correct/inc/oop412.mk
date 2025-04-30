#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop412  ########


oop412: run
	

build:  $(SRC)/oop412.f90
	-$(RM) oop412.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop412.f90 -o oop412.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop412.$(OBJX) check.$(OBJX) $(LIBS) -o oop412.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop412
	oop412.$(EXESUFFIX)

verify: ;

