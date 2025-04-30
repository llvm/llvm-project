#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test oop355  ########


oop355: run


build:  $(SRC)/oop355.f08
	-$(RM) oop355.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/oop355.f08 -o oop355.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) oop355.$(OBJX) check.$(OBJX) $(LIBS) -o oop355.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test oop355
	oop355.$(EXESUFFIX)

verify: ;

