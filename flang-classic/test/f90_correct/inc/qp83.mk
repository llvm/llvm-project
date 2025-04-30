#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtodble ########


qp83: run


build:  $(SRC)/qp83.f08
	-$(RM) qp83.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp83.f08 -o qp83.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp83.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qp83.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp83
	qp83.$(EXESUFFIX)

verify: ;


