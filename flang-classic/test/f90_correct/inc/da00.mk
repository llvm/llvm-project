#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test da00  ########


da00: run
	

build:  $(SRC)/da00.f
	-$(RM) da00.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/da00.f -o da00.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) da00.$(OBJX) check.$(OBJX) $(LIBS) -o da00.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test da00
	da00.$(EXESUFFIX)

verify: ;

da00.run: run

