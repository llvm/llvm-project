#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test fc70  ########


fc70: run
	

build:  $(SRC)/fc70.f
	-$(RM) fc70.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/fc70.f -o fc70.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) fc70.$(OBJX) check.$(OBJX) $(LIBS) -o fc70.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test fc70
	fc70.$(EXESUFFIX)

verify: ;

fc70.run: run

