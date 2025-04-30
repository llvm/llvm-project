#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test ls01  ########


ls01: run
	

build:  $(SRC)/ls01.f
	-$(RM) ls01.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/ls01.f -o ls01.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) ls01.$(OBJX) check.$(OBJX) $(LIBS) -o ls01.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test ls01
	ls01.$(EXESUFFIX)

verify: ;

ls01.run: run

