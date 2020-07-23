#===----------------------------------------------------------------------===##
#
# Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
#===----------------------------------------------------------------------===##

def _getSubstitution(substitution, config):
  for (orig, replacement) in config.substitutions:
    if orig == substitution:
      return replacement
  raise ValueError('Substitution {} is not in the config.'.format(substitution))

def configure(parameters, features, config, lit_config):
  # Apply parameters to the configuration first, since parameters are things
  # that we request explicitly and which might influence what features are
  # implicitly made available next.
  for param in parameters:
    feature = param.getFeature(config, lit_config.params)
    if feature:
      feature.enableIn(config)
      lit_config.note("Enabling Lit feature '{}' as a result of parameter '{}'".format(feature.getName(config), param.name))

  # Then, apply the automatically-detected features.
  printFeatures = []
  for feature in features:
    if feature.isSupported(config):
      feature.enableIn(config)
      printFeatures.append(feature.getName(config))
  printFeatures = ["'{}'".format(f) for f in sorted(printFeatures)]
  lit_config.note("Enabling implicitly detected Lit features {}".format(', '.join(printFeatures)))

  # Print the basic substitutions
  for sub in ('%{cxx}', '%{flags}', '%{compile_flags}', '%{link_flags}', '%{exec}'):
    lit_config.note("Using {} substitution: '{}'".format(sub, _getSubstitution(sub, config)))
