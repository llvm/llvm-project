#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dd00  ########


dd00: run
	

build:  $(SRC)/dd00.f
	-$(RM) dd00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dd00.f -o dd00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dd00.$(OBJX) check.$(OBJX) $(LIBS) -o dd00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dd00
	dd00.$(EXESUFFIX)

verify: ;

dd00.run: run

