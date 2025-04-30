#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

########## Make rule for test qtoint8  ########


qp64: run


build:  $(SRC)/qp64.f08
	-$(RM) qp64.$(EXESUFFIX) core *.d *.mod FOR*.DAT FTN* ftn* fort.*
	@echo ------------------------------------ building test $@
	-$(FC) -c $(FFLAGS) $(SRC)/check_mod.F90 -o check_mod.$(OBJX)
	-$(FC) -c $(FFLAGS) $(LDFLAGS) $(SRC)/qp64.f08 -o qp64.$(OBJX)
	-$(FC) $(FFLAGS) $(LDFLAGS) qp64.$(OBJX) check_mod.$(OBJX) $(LIBS) -o qp64.$(EXESUFFIX)


run:
	@echo ------------------------------------ executing test qp64
	qp64.$(EXESUFFIX)

verify: ;


