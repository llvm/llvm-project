#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test dc00  ########


dc00: run
	

build:  $(SRC)/dc00.f
	-$(RM) dc00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/dc00.f -o dc00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) dc00.$(OBJX) check.$(OBJX) $(LIBS) -o dc00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test dc00
	dc00.$(EXESUFFIX)

verify: ;

dc00.run: run

