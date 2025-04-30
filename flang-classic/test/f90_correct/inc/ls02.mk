#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls02  ########


ls02: run
	

build:  $(SRC)/ls02.f
	-$(RM) ls02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls02.f -o ls02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls02.$(OBJX) check.$(OBJX) $(LIBS) -o ls02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls02
	ls02.$(EXESUFFIX)

verify: ;

ls02.run: run

