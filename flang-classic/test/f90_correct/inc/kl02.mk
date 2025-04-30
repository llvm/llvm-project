#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test kl02  ########


kl02: run
	

build:  $(SRC)/kl02.f
	-$(RM) kl02.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(CC) -c $(CFLAGS) $(SRC)/check.c -o check.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/kl02.f -o kl02.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) kl02.$(OBJX) check.$(OBJX) $(LIBS) -o kl02.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test kl02
	kl02.$(EXESUFFIX)

verify: ;

kl02.run: run

