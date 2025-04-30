#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ha00  ########


ha00: run
	

build:  $(SRC)/ha00.f
	-$(RM) ha00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ha00.f -o ha00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ha00.$(OBJX) check.$(OBJX) $(LIBS) -o ha00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ha00
	ha00.$(EXESUFFIX)

verify: ;

ha00.run: run

