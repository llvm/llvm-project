# Premerge Infrastructure

This folder contains the terraform configuration files that define the GCP
resources used to run the premerge checks. Currently, only Google employees
with access to the GCP project where these checks are hosted are able to apply
changes. Pull requests from anyone are still welcome.

## Setup

- install terraform (https://developer.hashicorp.com/terraform/install?product_intent=terraform)
- get the GCP tokens: `gcloud auth application-default login`
- initialize terraform: `terraform init`

To apply any changes to the cluster:
- setup the cluster: `terraform apply`
- terraform will list the list of proposed changes.
- enter 'yes' when prompted.

## Setting the cluster up for the first time

```
terraform apply -target google_container_node_pool.llvm_premerge_linux_service
terraform apply -target google_container_node_pool.llvm_premerge_linux
terraform apply -target google_container_node_pool.llvm_premerge_windows
terraform apply
```

Setting the cluster up for the first time is more involved as there are certain
resources where terraform is unable to handle explicit dependencies. This means
that we have to set up the GKE cluster before we setup any of the Kubernetes
resources as otherwise the Terraform Kubernetes provider will error out.
