#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
build: atomic_update.$(OBJX)

run:
	@echo ------------ executing test $@
	-$(RUN2) ./atomic_update.$(EXESUFFIX) $(LOG)

verify: ;

atomic_update.$(OBJX): $(SRC)/atomic_update.f check.$(OBJX)
	@echo ------------ building test $@
	-$(FC) $(FFLAGS) $(SRC)/atomic_update.f
	@$(RM) ./a.$(EXESUFFIX)
	-$(FC) $(LDFLAGS) atomic_update.$(OBJX) check.$(OBJX) $(LIBS) -o atomic_update.$(EXESUFFIX)

